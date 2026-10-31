"""Tests verifying Docker containerization and Helm chart deployment manifests."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
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
        [helm_bin, "lint", str(CHART_DIR)],
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
