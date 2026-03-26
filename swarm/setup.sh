#!/bin/bash
# JARVIS v7 Setup & Deployment Helper
# Usage: bash jarvis_setup.sh [server|client]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check Python version
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    print_success "Python $PYTHON_VERSION found"
}

# Setup Server
setup_server() {
    print_header "JARVIS v7 Server Setup"
    
    check_python
    
    # Check for orchestrator
    print_info "Checking for OrchestratorV2_1..."
    if ! python3 -c "from orchestrator_v2_1 import OrchestratorV2_1" 2>/dev/null; then
        print_warning "OrchestratorV2_1 not found. Make sure it's in the parent directory."
        echo "Expected: ../orchestrator_v2_1.py"
    else
        print_success "OrchestratorV2_1 found"
    fi
    
    # Install dependencies
    print_info "Installing Python dependencies..."
    pip install flask flask-cors requests -q
    if [ $? -eq 0 ]; then
        print_success "Dependencies installed"
    else
        print_error "Failed to install dependencies"
        exit 1
    fi
    
    # Create results directory
    mkdir -p ./swarm_results
    print_success "Created results directory: ./swarm_results"
    
    # Test imports
    print_info "Testing imports..."
    python3 << 'EOF'
try:
    from flask import Flask
    from flask_cors import CORS
    print("✓ Flask dependencies OK")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)
EOF
    
    print_header "Server Setup Complete"
    echo "To start the server, run:"
    echo -e "  ${YELLOW}python3 swarm_api_server.py --port 5000${NC}"
    echo ""
    echo "Or with SearXNG:"
    echo -e "  ${YELLOW}python3 swarm_api_server.py --port 5000 --searxng http://localhost:8888${NC}"
    echo ""
    echo "Or set environment variables:"
    echo -e "  ${YELLOW}export SWARM_API_PORT=5000${NC}"
    echo -e "  ${YELLOW}export SEARXNG_URL=http://localhost:8888${NC}"
    echo -e "  ${YELLOW}python3 swarm_api_server.py${NC}"
}

# Setup Client
setup_client() {
    print_header "JARVIS v7 Client Setup"
    
    check_python
    
    # Check for existing dependencies
    print_info "Checking existing dependencies..."
    python3 << 'EOF'
missing = []
deps = ['sounddevice', 'soundfile', 'numpy', 'whisper', 'pynput', 'requests', 'beautifulsoup4']
for dep in deps:
    try:
        __import__(dep.replace('-', '_'))
        print(f"✓ {dep}")
    except ImportError:
        print(f"✗ {dep}")
        missing.append(dep)

if missing:
    print(f"\nMissing: {', '.join(missing)}")
    print("Install with: pip install " + " ".join(missing))
EOF
    
    # Check configuration
    print_info "Checking configuration..."
    if grep -q "SWARM_SERVER" jarvis_with_deepsearch.py; then
        print_success "SWARM_SERVER found in config"
    else
        print_error "SWARM_SERVER not found. Check configuration."
    fi
    
    # Ask for server IP
    read -p "Enter your Swarm server IP (default: 10.0.0.58): " SERVER_IP
    SERVER_IP="${SERVER_IP:-10.0.0.58}"
    
    read -p "Enter your Swarm server port (default: 5000): " SERVER_PORT
    SERVER_PORT="${SERVER_PORT:-5000}"
    
    # Update configuration
    print_info "Updating configuration..."
    sed -i.bak "s|SWARM_SERVER = \".*\"|SWARM_SERVER = \"http://$SERVER_IP:$SERVER_PORT\"|" jarvis_with_deepsearch.py
    print_success "Configuration updated"
    
    # Test connectivity
    print_info "Testing server connectivity..."
    if curl -s http://$SERVER_IP:$SERVER_PORT/health > /dev/null 2>&1; then
        print_success "Server is reachable"
    else
        print_warning "Could not reach server at http://$SERVER_IP:$SERVER_PORT"
        echo "Make sure the Swarm server is running:"
        echo -e "  ${YELLOW}python3 swarm_api_server.py --port $SERVER_PORT${NC}"
    fi
    
    print_header "Client Setup Complete"
    echo "Configuration:"
    echo "  Server: http://$SERVER_IP:$SERVER_PORT"
    echo ""
    echo "To start JARVIS, run:"
    echo -e "  ${YELLOW}export TAVILY_API_KEY='tvly-xxx'${NC}  # Optional"
    echo -e "  ${YELLOW}python3 jarvis_with_deepsearch.py${NC}"
}

# Test API
test_api() {
    print_header "Testing Swarm API"
    
    read -p "Enter server IP (default: 10.0.0.58): " SERVER_IP
    SERVER_IP="${SERVER_IP:-10.0.0.58}"
    
    read -p "Enter server port (default: 5000): " SERVER_PORT
    SERVER_PORT="${SERVER_PORT:-5000}"
    
    URL="http://$SERVER_IP:$SERVER_PORT"
    
    print_info "Testing connectivity to $URL"
    
    # Health check
    echo ""
    print_info "1. Health check..."
    if RESPONSE=$(curl -s $URL/health); then
        print_success "Server responded"
        echo "   Response: $RESPONSE"
    else
        print_error "No response from server"
        exit 1
    fi
    
    # Status check
    echo ""
    print_info "2. Status check..."
    if RESPONSE=$(curl -s $URL/status); then
        print_success "Status available"
        echo "   Response: $RESPONSE"
    else
        print_error "Could not fetch status"
    fi
    
    # Test sync query
    echo ""
    print_info "3. Testing sync query (this will take a while)..."
    echo "   Send simple query: 'What is 2+2?'"
    read -p "   Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        RESPONSE=$(curl -s -X POST $URL/query \
          -H "Content-Type: application/json" \
          -d '{"question": "What is 2+2?"}' \
          --max-time 300)
        
        if echo $RESPONSE | grep -q '"answer"'; then
            print_success "Sync query works"
            echo "   Response: ${RESPONSE:0:100}..."
        else
            print_error "Sync query failed"
            echo "   Response: $RESPONSE"
        fi
    fi
    
    # Test async query
    echo ""
    print_info "4. Testing async query..."
    RESPONSE=$(curl -s -X POST $URL/query_async \
      -H "Content-Type: application/json" \
      -d '{"question": "What is artificial intelligence?"}')
    
    if echo $RESPONSE | grep -q '"job_id"'; then
        print_success "Async query works"
        JOB_ID=$(echo $RESPONSE | grep -o '"job_id":"[^"]*"' | cut -d'"' -f4)
        echo "   Job ID: $JOB_ID"
        
        # Check result
        echo ""
        print_info "5. Checking job result..."
        sleep 2
        RESULT=$(curl -s $URL/result/$JOB_ID)
        echo "   Status: $(echo $RESULT | grep -o '"status":"[^"]*"' | cut -d'"' -f4)"
    else
        print_error "Async query failed"
        echo "   Response: $RESPONSE"
    fi
    
    print_header "API Tests Complete"
}

# Show help
show_help() {
    echo "JARVIS v7 Setup Helper"
    echo ""
    echo "Usage: bash jarvis_setup.sh [command]"
    echo ""
    echo "Commands:"
    echo "  server     Setup Swarm API server"
    echo "  client     Setup JARVIS client"
    echo "  test       Test API connectivity"
    echo "  help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  bash jarvis_setup.sh server     # Run on server machine"
    echo "  bash jarvis_setup.sh client     # Run on client machine"
    echo "  bash jarvis_setup.sh test       # Test connection"
}

# Main
case "${1:-help}" in
    server)
        setup_server
        ;;
    client)
        setup_client
        ;;
    test)
        test_api
        ;;
    help)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
