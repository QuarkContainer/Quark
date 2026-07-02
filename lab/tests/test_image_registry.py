"""Tests for Keska mirror auth shell generation."""

from keska_lab.setup.image_registry import (
    DEFAULT_IMAGE_REGISTRY,
    GCLOUD_AUTH_LOGIN_CMD,
    check_registry_auth_ctr_script,
    check_registry_auth_script,
    ctr_pull_with_mirror_script,
    docker_auth_preamble,
    docker_pull_with_mirror_script,
    gcloud_auth_login_script,
    gcloud_docker_login_script,
    lab_path_setup_script,
    registry_auth_diagnostic_script,
)


def test_check_script_uses_login_path_and_oauth_login():
    script = check_registry_auth_script(DEFAULT_IMAGE_REGISTRY)
    assert "google-cloud-sdk/bin" in script
    assert "oauth2accesstoken" in script
    assert "docker login" in script
    assert "docker pull" in script
    assert "<<PY" not in script


def test_docker_pull_includes_auth_preamble():
    script = docker_pull_with_mirror_script("busybox", DEFAULT_IMAGE_REGISTRY)
    assert "oauth2accesstoken" in script
    assert "print-access-token" in script or "gcloud auth configure-docker" in script


def test_ctr_pull_docker_fallback_includes_auth():
    script = ctr_pull_with_mirror_script("busybox", DEFAULT_IMAGE_REGISTRY, snapshotter="devmapper")
    assert "ctr images import" in script
    assert "oauth2accesstoken" in script
    assert "ctr_pull_ref" in script
    assert "unpack=${dm_src:-$canonical}" in script


def test_gcloud_login_script_has_host():
    script = gcloud_docker_login_script("europe-north1-docker.pkg.dev")
    assert "europe-north1-docker.pkg.dev" in script
    assert lab_path_setup_script() in docker_auth_preamble(DEFAULT_IMAGE_REGISTRY)


def test_gcloud_auth_login_uses_no_launch_browser():
    script = gcloud_auth_login_script()
    assert GCLOUD_AUTH_LOGIN_CMD in script
    assert "--no-launch-browser" in script
    assert lab_path_setup_script() in script
    diag = registry_auth_diagnostic_script(DEFAULT_IMAGE_REGISTRY)
    assert "--no-launch-browser" in diag


def test_ctr_auth_uses_user_password_not_secret():
    script = ctr_pull_with_mirror_script("busybox", DEFAULT_IMAGE_REGISTRY, snapshotter="devmapper")
    assert 'oauth2accesstoken:$token' in script
    assert "--secret" not in script
    ctr_probe = check_registry_auth_ctr_script(DEFAULT_IMAGE_REGISTRY)
    assert '-u "oauth2accesstoken:$token"' in ctr_probe
    assert "--secret" not in ctr_probe
