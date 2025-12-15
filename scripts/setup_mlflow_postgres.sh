#!/bin/bash
# =============================================================================
# MLflow Server with PostgreSQL Backend Setup Script
# =============================================================================
# This script sets up a PostgreSQL database in Docker and configures MLflow
# server to use it as the backend store for experiment tracking.
#
# Usage:
#   ./scripts/setup_mlflow_postgres.sh [start|stop|restart|status|logs]
#
# Default action is 'start' if no argument provided.
# =============================================================================

set -euo pipefail

# Configuration
BASE_DIR="/media/aglisman/Data/models"
mkdir -p "$BASE_DIR"

POSTGRES_CONTAINER="mlflow-postgres"
POSTGRES_DB="mlflow"
POSTGRES_USER="mlflow"
POSTGRES_PASSWORD="mlflow_password_2024"
POSTGRES_PORT="5434"
POSTGRES_DATA_DIR="${BASE_DIR}/mlflow-postgres"

MLFLOW_PORT="8084"
MLFLOW_ARTIFACTS_PATH="${BASE_DIR}/mlflow-artifacts"
MLFLOW_PID_FILE="/tmp/mlflow_server.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
  echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
  echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
  echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
  if ! docker info >/dev/null 2>&1; then
    log_error "Docker is not running. Please start Docker first."
    exit 1
  fi
}

# Create directories
setup_directories() {
  log_info "Setting up directories..."
  mkdir -p "$POSTGRES_DATA_DIR"
  mkdir -p "$MLFLOW_ARTIFACTS_PATH"
  log_success "Directories created"
}

# Start PostgreSQL container
start_postgres() {
  log_info "Starting PostgreSQL container..."
  local needs_create=0
  local mapped_port=""

  if docker inspect "$POSTGRES_CONTAINER" >/dev/null 2>&1; then
    mapped_port=$(docker inspect --format '{{with (index .HostConfig.PortBindings "5432/tcp")}}{{(index . 0).HostPort}}{{end}}' "$POSTGRES_CONTAINER" 2>/dev/null || echo "")

    if [[ -z "$mapped_port" || "$mapped_port" != "$POSTGRES_PORT" ]]; then
      log_warning "Existing PostgreSQL container missing port mapping $POSTGRES_PORT (found ${mapped_port:-none}). Recreating container..."
      docker rm -f "$POSTGRES_CONTAINER"
      needs_create=1
    fi
  else
    needs_create=1
  fi

  if [[ $needs_create -eq 1 ]]; then
    log_info "Creating new PostgreSQL container..."
    docker run -d \
      --name "$POSTGRES_CONTAINER" \
      --restart unless-stopped \
      -e POSTGRES_DB="$POSTGRES_DB" \
      -e POSTGRES_USER="$POSTGRES_USER" \
      -e POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
      -p "$POSTGRES_PORT:5432" \
      -v "$POSTGRES_DATA_DIR:/var/lib/postgresql/data" \
      postgres:15-alpine
  else
    local is_running
    is_running=$(docker inspect --format '{{.State.Running}}' "$POSTGRES_CONTAINER")

    if [[ "$is_running" == "true" ]]; then
      log_warning "PostgreSQL container is already running"
      return 0
    else
      log_info "Starting existing PostgreSQL container..."
      docker start "$POSTGRES_CONTAINER"
    fi
  fi

  # Wait for PostgreSQL to be ready
  log_info "Waiting for PostgreSQL to be ready..."
  max_attempts=30
  attempt=1

  while [ $attempt -le $max_attempts ]; do
    if docker exec "$POSTGRES_CONTAINER" pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null 2>&1; then
      log_success "PostgreSQL is ready!"
      break
    fi

    if [ $attempt -eq $max_attempts ]; then
      log_error "PostgreSQL failed to start within $max_attempts attempts"
      return 1
    fi

    log_info "Attempt $attempt/$max_attempts: PostgreSQL not ready yet, waiting..."
    sleep 2
    ((attempt++))
  done
}

# Install Python dependencies
install_dependencies() {
  log_info "Installing Python dependencies..."

  # Check if we're in a virtual environment
  if [[ "${VIRTUAL_ENV:-}" != "" ]]; then
    log_info "Using virtual environment: $VIRTUAL_ENV"
  else
    log_warning "No virtual environment detected. Consider using one."
  fi

  # Install required packages
  uv pip install psycopg2-binary mlflow
  log_success "Dependencies installed"
}

# Start MLflow server
start_mlflow() {
  log_info "Starting MLflow server..."

  # Check if MLflow server is already running
  if [[ -f "$MLFLOW_PID_FILE" ]] && kill -0 "$(cat "$MLFLOW_PID_FILE")" 2>/dev/null; then
    log_warning "MLflow server is already running (PID: $(cat "$MLFLOW_PID_FILE"))"
    return 0
  fi

  # Database URL
  DB_URL="postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@localhost:$POSTGRES_PORT/$POSTGRES_DB"
  log_info "Using database URL: $DB_URL"

  # Start MLflow server in background
  nohup mlflow server \
    --backend-store-uri "$DB_URL" \
    --default-artifact-root "file://$MLFLOW_ARTIFACTS_PATH" \
    --host "0.0.0.0" \
    --workers "12" \
    --port "$MLFLOW_PORT" \
    --serve-artifacts \
    >"/tmp/mlflow_server.log" 2>&1 &

  # Save PID
  echo $! >"$MLFLOW_PID_FILE"

  # Wait for MLflow to be ready
  log_info "Waiting for MLflow server to be ready..."
  max_attempts=30
  attempt=1

  while [ $attempt -le $max_attempts ]; do
    if curl -s "http://localhost:$MLFLOW_PORT/health" >/dev/null 2>&1; then
      log_success "MLflow server is ready!"
      break
    fi

    if [ $attempt -eq $max_attempts ]; then
      log_error "MLflow server failed to start within $max_attempts attempts"
      return 1
    fi

    log_info "Attempt $attempt/$max_attempts: MLflow not ready yet, waiting..."
    sleep 2
    ((attempt++))
  done

  log_success "MLflow server started (PID: $(cat "$MLFLOW_PID_FILE"))"
  log_info "MLflow UI available at: http://localhost:$MLFLOW_PORT"
}

# Stop services
stop_services() {
  log_info "Stopping services..."

  # Stop MLflow server
  if [[ -f "$MLFLOW_PID_FILE" ]] && kill -0 "$(cat "$MLFLOW_PID_FILE")" 2>/dev/null; then
    log_info "Stopping MLflow server..."
    kill "$(cat "$MLFLOW_PID_FILE")"
    rm -f "$MLFLOW_PID_FILE"
    log_success "MLflow server stopped"
  else
    log_warning "MLflow server not running"
  fi

  # Stop PostgreSQL container
  if docker ps --format 'table {{.Names}}' | grep -q "^${POSTGRES_CONTAINER}$"; then
    log_info "Stopping PostgreSQL container..."
    docker stop "$POSTGRES_CONTAINER"
    log_success "PostgreSQL container stopped"
  else
    log_warning "PostgreSQL container not running"
  fi
}

# Show status
show_status() {
  log_info "Service Status:"
  echo ""

  # PostgreSQL status
  if docker ps --format 'table {{.Names}}' | grep -q "^${POSTGRES_CONTAINER}$"; then
    log_success "PostgreSQL: Running"
    echo "  Container: $POSTGRES_CONTAINER"
    echo "  Port: $POSTGRES_PORT"
    echo "  Database: $POSTGRES_DB"
  else
    log_error "PostgreSQL: Not running"
  fi

  echo ""

  # MLflow status
  if [[ -f "$MLFLOW_PID_FILE" ]] && kill -0 "$(cat "$MLFLOW_PID_FILE")" 2>/dev/null; then
    log_success "MLflow Server: Running (PID: $(cat "$MLFLOW_PID_FILE"))"
    echo "  URL: http://localhost:$MLFLOW_PORT"
    echo "  Artifacts: $MLFLOW_ARTIFACTS_PATH"
    echo "  Backend: PostgreSQL"
  else
    log_error "MLflow Server: Not running"
  fi

  echo ""
  log_info "Connection string for your config:"
  echo "  mlflow_tracking_uri: \"http://127.0.0.1:$MLFLOW_PORT\""
}

# Show logs
show_logs() {
  log_info "Service Logs:"
  echo ""

  if [[ "$1" == "postgres" ]] || [[ "$1" == "all" ]]; then
    log_info "PostgreSQL logs (last 20 lines):"
    docker logs --tail 20 "$POSTGRES_CONTAINER" 2>/dev/null || log_warning "No PostgreSQL logs available"
    echo ""
  fi

  if [[ "$1" == "mlflow" ]] || [[ "$1" == "all" ]]; then
    log_info "MLflow logs (last 20 lines):"
    if [[ -f "/tmp/mlflow_server.log" ]]; then
      tail -20 "/tmp/mlflow_server.log"
    else
      log_warning "No MLflow logs available"
    fi
    echo ""
  fi
}

# Main function
main() {
  local action="${1:-start}"

  case "$action" in
  start)
    log_info "Starting MLflow with PostgreSQL backend..."
    check_docker
    setup_directories
    start_postgres
    install_dependencies
    start_mlflow
    show_status
    ;;
  stop)
    stop_services
    ;;
  restart)
    stop_services
    sleep 2
    main start
    ;;
  status)
    show_status
    ;;
  logs)
    show_logs "${2:-all}"
    ;;
  *)
    echo "Usage: $0 [start|stop|restart|status|logs [postgres|mlflow|all]]"
    echo ""
    echo "Commands:"
    echo "  start   - Start PostgreSQL and MLflow server (default)"
    echo "  stop    - Stop both services"
    echo "  restart - Restart both services"
    echo "  status  - Show service status"
    echo "  logs    - Show service logs"
    echo ""
    echo "Examples:"
    echo "  $0                    # Start services"
    echo "  $0 start              # Start services"
    echo "  $0 stop               # Stop services"
    echo "  $0 logs mlflow        # Show MLflow logs only"
    exit 1
    ;;
  esac
}

# Run main function with all arguments
main "$@"
