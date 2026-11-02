{{/*
Expand the name of the chart.
*/}}
{{- define "sportsagentic-agent.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "sportsagentic-agent.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "sportsagentic-agent.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "sportsagentic-agent.labels" -}}
helm.sh/chart: {{ include "sportsagentic-agent.chart" . }}
{{ include "sportsagentic-agent.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "sportsagentic-agent.selectorLabels" -}}
app.kubernetes.io/name: {{ include "sportsagentic-agent.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "sportsagentic-agent.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "sportsagentic-agent.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the Secret name holding GEMINI_API_KEY. Fails when no key was provided,
because failover would be off and /health wouldn't show it. Only call this when
secrets.geminiDisabled is unset.
*/}}
{{- define "sportsagentic-agent.secretName" -}}
{{- if .Values.secrets.existingSecret }}
{{- .Values.secrets.existingSecret }}
{{- else if trim (.Values.secrets.geminiApiKey | default "") }}
{{- include "sportsagentic-agent.fullname" . }}
{{- else }}
{{- fail "Gemini API key is missing: set secrets.existingSecret to a Secret holding it, set secrets.geminiApiKey to a non-empty value, or set secrets.geminiDisabled=true to run without Gemini." }}
{{- end }}
{{- end }}

{{/*
Return the image tag as a string. A number from a values file or --reuse-values
is a float64 here, and toString would turn 8542126 into 8.542126e+06.
*/}}
{{- define "sportsagentic-agent.imageTag" -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion -}}
{{- if kindIs "float64" $tag -}}
{{- if ne (floor $tag) $tag -}}
{{- fail (printf "image.tag %v is not a whole number, so YAML has already changed it. Quote it, or pass --set-string image.tag=<tag>." $tag) -}}
{{- end -}}
{{- printf "%.0f" $tag -}}
{{- else -}}
{{- $tag -}}
{{- end -}}
{{- end }}
