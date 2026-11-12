"""Tests verifying Docker containerization and Helm chart deployment manifests."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import pytest
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
CHART_DIR = ROOT_DIR / "charts" / "sportsagentic-agent"


def test_dockerfile_and_compose_exist_and_valid() -> None:
    """Verify Dockerfile, .dockerignore, and docker-compose.yml are present and well-formed."""
    dockerfile = ROOT_DIR / "Dockerfile"
    dockerignore = ROOT_DIR / ".dockerignore"
    compose_file = ROOT_DIR / "docker-compose.yml"

    assert dockerfile.exists()
    assert dockerignore.exists()
    assert compose_file.exists()

    dockerfile_text = dockerfile.read_text(encoding="utf-8")
    assert "uvicorn app.main:app" in dockerfile_text
    assert "USER appuser" in dockerfile_text
    assert "EXPOSE 8000" in dockerfile_text

    compose_data = yaml.safe_load(compose_file.read_text(encoding="utf-8"))
    service = compose_data["services"]["sportsagentic-agent"]
    assert ":8000" in service["ports"][0]


def test_helm_chart_structure_and_values() -> None:
    """Verify Chart.yaml, values.yaml, and required Kubernetes templates exist."""
    chart_yaml = yaml.safe_load((CHART_DIR / "Chart.yaml").read_text(encoding="utf-8"))
    assert chart_yaml["apiVersion"] == "v2"
    assert chart_yaml["name"] == "sportsagentic-agent"

    values = yaml.safe_load((CHART_DIR / "values.yaml").read_text(encoding="utf-8"))
    assert values["config"]["mcpServerUrl"] == "http://sportsagentic:8080/sse"
    assert values["config"]["llamaServerUrl"] == "http://llama-server:8081/v1"
    assert values["config"]["defaultLlmProvider"] == "auto"
    assert values["service"]["port"] == 8000

    expected_templates = [
        "_helpers.tpl",
        "configmap.yaml",
        "secret.yaml",
        "serviceaccount.yaml",
        "deployment.yaml",
        "service.yaml",
        "ingress.yaml",
        "hpa.yaml",
        "NOTES.txt",
    ]
    for tpl in expected_templates:
        assert (CHART_DIR / "templates" / tpl).exists(), f"Missing template {tpl}"


def test_helm_lint_and_template_rendering() -> None:
    """Run helm lint and helm template if the helm binary is installed."""
    helm_bin = shutil.which("helm")
    if not helm_bin:
        return

    lint_proc = subprocess.run(
        [helm_bin, "lint", str(CHART_DIR), "--set", "secrets.geminiApiKey=test-secret-key"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert lint_proc.returncode == 0, f"helm lint failed: {lint_proc.stdout}\n{lint_proc.stderr}"

    tpl_proc = subprocess.run(
        [
            helm_bin,
            "template",
            "test-release",
            str(CHART_DIR),
            "--set",
            "secrets.geminiApiKey=test-secret-key",
            "--set",
            "ingress.enabled=true",
            "--set",
            "autoscaling.enabled=true",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert tpl_proc.returncode == 0, f"helm template failed: {tpl_proc.stderr}"

    docs = [d for d in yaml.safe_load_all(tpl_proc.stdout) if d]
    kinds = {d["kind"] for d in docs}
    assert {
        "ServiceAccount",
        "Secret",
        "ConfigMap",
        "Service",
        "Deployment",
        "Ingress",
        "HorizontalPodAutoscaler",
    }.issubset(kinds)


def render_chart(*args: str) -> subprocess.CompletedProcess[str]:
    """Run helm template on the chart with the given extra arguments."""
    helm_bin = shutil.which("helm")
    assert helm_bin
    return subprocess.run(
        [helm_bin, "template", "test-release", str(CHART_DIR), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def deployment_container(rendered: str) -> dict:
    """Return the agent container from rendered manifests."""
    docs = [d for d in yaml.safe_load_all(rendered) if d]
    deployment = next(d for d in docs if d["kind"] == "Deployment")
    return deployment["spec"]["template"]["spec"]["containers"][0]


@pytest.mark.skipif(not shutil.which("helm"), reason="helm is not installed")
@pytest.mark.parametrize(
    "args",
    [
        (),
        ("--set", "secrets.geminiApiKey="),
        ("--set", "secrets.geminiApiKey=   "),
        ("--set", "secrets.existingSecret=,secrets.geminiApiKey="),
    ],
)
def test_helm_fails_without_a_gemini_key(args: tuple[str, ...]) -> None:
    """An empty key must stop the release instead of disabling failover silently."""
    proc = render_chart(*args)
    assert proc.returncode != 0
    assert "Gemini API key is missing" in proc.stderr


@pytest.mark.skipif(not shutil.which("helm"), reason="helm is not installed")
def test_helm_existing_secret_is_required_by_the_pod() -> None:
    """A provided key must not be optional, so a bad Secret stops the rollout."""
    proc = render_chart("--set", "secrets.existingSecret=agent-secrets")
    assert proc.returncode == 0, proc.stderr
    ref = deployment_container(proc.stdout)["env"][0]["valueFrom"]["secretKeyRef"]
    assert ref == {"name": "agent-secrets", "key": "GEMINI_API_KEY"}
    assert "kind: Secret" not in proc.stdout


@pytest.mark.skipif(not shutil.which("helm"), reason="helm is not installed")
def test_helm_gemini_can_be_disabled_explicitly() -> None:
    """Running without Gemini is a stated choice that leaves no key behind."""
    proc = render_chart("--set", "secrets.geminiDisabled=true")
    assert proc.returncode == 0, proc.stderr
    assert "env" not in deployment_container(proc.stdout)
    assert "kind: Secret" not in proc.stdout


@pytest.mark.skipif(not shutil.which("helm"), reason="helm is not installed")
def test_helm_numeric_image_tag_stays_a_string(tmp_path: Path) -> None:
    """A numeric SHA must keep its digits however it reaches the chart."""
    values = tmp_path / "values.yaml"
    values.write_text("image:\n  tag: 8542126\n", encoding="utf-8")
    for args in (("--set", "image.tag=8542126"), ("-f", str(values))):
        proc = render_chart("--set", "secrets.existingSecret=agent-secrets", *args)
        assert proc.returncode == 0, proc.stderr
        assert deployment_container(proc.stdout)["image"].endswith(":8542126")


@pytest.mark.skipif(not shutil.which("helm"), reason="helm is not installed")
def test_helm_fails_on_a_mangled_image_tag(tmp_path: Path) -> None:
    """A tag YAML has already rounded can't be recovered, so it must fail."""
    values = tmp_path / "values.yaml"
    values.write_text("image:\n  tag: 1.10\n", encoding="utf-8")
    proc = render_chart("--set", "secrets.existingSecret=agent-secrets", "-f", str(values))
    assert proc.returncode != 0
    assert "image.tag" in proc.stderr


@pytest.mark.skipif(not shutil.which("helm"), reason="helm is not installed")
def test_helm_telemetry_env_defaults() -> None:
    """KAFKA_BOOTSTRAP comes from the ConfigMap and TELEMETRY_SALT from an optional Secret."""
    proc = render_chart("--set", "secrets.existingSecret=agent-secrets")
    assert proc.returncode == 0, proc.stderr
    docs = [d for d in yaml.safe_load_all(proc.stdout) if d]
    configmap = next(d for d in docs if d["kind"] == "ConfigMap")
    assert configmap["data"]["KAFKA_BOOTSTRAP"] == "sa-kafka-kafka-bootstrap.data:9092"
    container = deployment_container(proc.stdout)
    assert {"secretRef": {"name": "telemetry-salt", "optional": True}} in container["envFrom"]
    assert container["env"][0]["name"] == "GEMINI_API_KEY"


@pytest.mark.skipif(not shutil.which("helm"), reason="helm is not installed")
@pytest.mark.parametrize("value", ["", "null"])
def test_helm_telemetry_can_be_left_out(value: str) -> None:
    """Empty or missing values (as with --reuse-values from an older release) render telemetry off."""
    proc = render_chart(
        "--set",
        "secrets.existingSecret=agent-secrets",
        "--set",
        f"config.kafkaBootstrap={value}",
        "--set",
        f"secrets.telemetrySaltSecret={value}",
    )
    assert proc.returncode == 0, proc.stderr
    docs = [d for d in yaml.safe_load_all(proc.stdout) if d]
    configmap = next(d for d in docs if d["kind"] == "ConfigMap")
    assert configmap["data"]["KAFKA_BOOTSTRAP"] == ""
    container = deployment_container(proc.stdout)
    assert all("secretRef" not in source for source in container["envFrom"])
