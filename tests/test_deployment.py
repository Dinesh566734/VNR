from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_exposes_api_and_model_runtime_contract() -> None:
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.11-slim" in dockerfile
    assert "EXPOSE 8000" in dockerfile
    assert "/app/models/sentinel_gat_best.pt" in dockerfile
    assert "src.inference.api:create_app" in dockerfile
    assert "--factory" in dockerfile


def test_docker_compose_defines_serving_and_observability_stack() -> None:
    compose = yaml.safe_load((ROOT / "docker" / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]

    assert {"sentinel-api", "redis-graph", "prometheus", "grafana"} <= set(services)
    assert services["sentinel-api"]["ports"] == ["8000:8000"]
    assert services["redis-graph"]["ports"] == ["6379:6379"]
    assert services["prometheus"]["ports"] == ["9090:9090"]
    assert services["grafana"]["ports"] == ["3000:3000"]


def test_prometheus_config_scrapes_the_api_metrics_endpoint() -> None:
    config = yaml.safe_load((ROOT / "docker" / "prometheus.yml").read_text(encoding="utf-8"))
    jobs = {job["job_name"]: job for job in config["scrape_configs"]}

    assert "sentinel-api" in jobs
    assert jobs["sentinel-api"]["metrics_path"] == "/metrics"
    assert jobs["sentinel-api"]["static_configs"][0]["targets"] == ["sentinel-api:8000"]


def test_kubernetes_deployment_matches_phase_8_serving_requirements() -> None:
    documents = list(
        yaml.safe_load_all((ROOT / "k8s" / "deployment.yaml").read_text(encoding="utf-8"))
    )
    service = next(document for document in documents if document["kind"] == "Service")
    deployment = next(document for document in documents if document["kind"] == "Deployment")
    container = deployment["spec"]["template"]["spec"]["containers"][0]

    assert service["spec"]["ports"][0]["port"] == 8000
    assert deployment["spec"]["replicas"] == 3
    assert deployment["spec"]["template"]["spec"]["nodeSelector"]["accelerator"] == "nvidia"
    assert container["resources"]["limits"]["cpu"] == "2"
    assert container["resources"]["limits"]["memory"] == "4Gi"
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert container["livenessProbe"]["periodSeconds"] == 30
    assert container["readinessProbe"]["httpGet"]["path"] == "/health"


def test_kubernetes_hpa_scales_on_cpu_and_queue_depth() -> None:
    hpa = yaml.safe_load((ROOT / "k8s" / "hpa.yaml").read_text(encoding="utf-8"))
    metrics = hpa["spec"]["metrics"]

    assert hpa["spec"]["minReplicas"] == 3
    assert hpa["spec"]["maxReplicas"] == 20
    assert any(
        metric.get("type") == "Resource"
        and metric["resource"]["name"] == "cpu"
        and metric["resource"]["target"]["averageUtilization"] == 70
        for metric in metrics
    )
    assert any(
        metric.get("type") == "Pods"
        and metric["pods"]["metric"]["name"] == "inference_queue_depth"
        and metric["pods"]["target"]["averageValue"] == "100"
        for metric in metrics
    )
