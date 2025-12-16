# MLflow PostgreSQL Docker Setup

This directory contains scripts for setting up MLflow with PostgreSQL backend using Docker.

## Quick Start

```bash
# Start services
./scripts/infra/setup_mlflow_postgres.sh

# Check status
./scripts/infra/setup_mlflow_postgres.sh status

# View logs
./scripts/infra/setup_mlflow_postgres.sh logs

# Stop services
./scripts/infra/setup_mlflow_postgres.sh stop
```

## What the Script Does

1. **PostgreSQL Container**: Creates a PostgreSQL 15 container with persistent data storage
2. **MLflow Server**: Starts MLflow server with PostgreSQL as backend store
3. **Dependencies**: Installs required Python packages (`psycopg2-binary`, `mlflow`)
4. **Health Checks**: Waits for services to be ready before proceeding

## Configuration

The script uses these default settings (modify in the script if needed):

```bash
POSTGRES_CONTAINER="mlflow-postgres"
POSTGRES_DB="mlflow"
POSTGRES_USER="mlflow"
POSTGRES_PASSWORD="mlflow_password_2024"
POSTGRES_PORT="5432"
POSTGRES_DATA_DIR="/media/aglisman/Data/postgres-mlflow"

MLFLOW_PORT="8080"
MLFLOW_ARTIFACTS_PATH="/media/aglisman/Data/models"
```

## Commands

- `start` - Start PostgreSQL and MLflow server (default)
- `stop` - Stop both services
- `restart` - Restart both services
- `status` - Show service status
- `logs [postgres|mlflow|all]` - Show service logs

## Integration with HPO Config

Your HPO config should use:

```yaml
mlflow_tracking_uri: "http://127.0.0.1:8084"
```

## Troubleshooting

### PostgreSQL Issues

```bash
# Check PostgreSQL logs
./scripts/infra/setup_mlflow_postgres.sh logs postgres

# Connect to database directly
docker exec -it mlflow-postgres psql -U mlflow -d mlflow
```

### MLflow Issues

```bash
# Check MLflow logs
./scripts/infra/setup_mlflow_postgres.sh logs mlflow

# Manual start (for debugging)
mlflow server --backend-store-uri postgresql://mlflow:mlflow_password_2024@localhost:5432/mlflow \
              --default-artifact-root file:///media/aglisman/Data/models \
              --host 0.0.0.0 --port 8080
```

### Port Conflicts

If ports 5432 or 8080 are in use, modify the script variables:

- `POSTGRES_PORT` - PostgreSQL port
- `MLFLOW_PORT` - MLflow server port

## Data Persistence

- **PostgreSQL data**: `/media/aglisman/Data/postgres-mlflow`
- **MLflow artifacts**: `/media/aglisman/Data/models`

Both locations persist data between container restarts.
