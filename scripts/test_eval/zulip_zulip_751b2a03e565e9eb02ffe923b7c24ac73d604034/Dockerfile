FROM zulip/ci:bullseye-6.0

WORKDIR /repo

CMD ["bash", "-c", "mkdir -p /srv/zulip-venv-cache && ./tools/ci/setup-backend --skip-dev-db-build && ./tools/test-backend --ban-console-output"]