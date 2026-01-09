#!/bin/bash
# NLHF Tournament Runner
#
# Usage:
#   ./run.sh              # Quick test with mock models  
#   ./run.sh api          # Compare OpenAI models with heuristic
#   ./run.sh api-llm      # Compare OpenAI models with LLM arbiter
#   ./run.sh config       # Use models.yaml config
#   ./run.sh full         # Full experiment (3 models)

set -e

# Load .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "✅ Loaded .env"
fi

# Check API key for non-mock modes
check_api_key() {
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "❌ OPENAI_API_KEY not set in .env"
        exit 1
    fi
    echo "✅ API key: ${OPENAI_API_KEY:0:10}..."
}

MODE="${1:-mock}"

echo "=================================================="
echo "NLHF Tournament - Mode: $MODE"
echo "=================================================="

case "$MODE" in
    mock)
        echo "🎭 Mock models + Heuristic arbiter (test)"
        python main.py \
            --prompts 10 \
            --matches 1 \
            --arbiter heuristic \
            --output ./output/mock
        ;;
    
    api)
        check_api_key
        echo "🌐 OpenAI models + Heuristic arbiter"
        python main.py \
            --models gpt-4o-mini gpt-3.5-turbo \
            --prompts 20 \
            --matches 2 \
            --arbiter heuristic \
            --output ./output/api
        ;;
    
    api-llm)
        check_api_key
        echo "🌐 OpenAI models + LLM arbiter"
        python main.py \
            --models gpt-4o-mini gpt-3.5-turbo \
            --prompts 10 \
            --matches 2 \
            --arbiter llm \
            --arbiter-model gpt-4o \
            --output ./output/api-llm
        ;;
    
    config)
        check_api_key
        echo "📄 Using config/models.yaml"
        python main.py \
            --config config/models.yaml \
            --prompts 20 \
            --matches 2 \
            --arbiter hybrid \
            --output ./output/config
        ;;

    full)
        check_api_key
        echo "🚀 Full experiment (3 models, LLM arbiter)"
        python main.py \
            --models gpt-4o gpt-4o-mini gpt-3.5-turbo \
            --prompts 30 \
            --matches 3 \
            --arbiter llm \
            --arbiter-model gpt-4o \
            --output ./output/full
        ;;
    
    *)
        echo "Unknown mode: $MODE"
        echo ""
        echo "Available modes:"
        echo "  mock      - Quick test with mock models"
        echo "  api       - OpenAI models + heuristic arbiter"
        echo "  api-llm   - OpenAI models + LLM arbiter"
        echo "  config    - Use config/models.yaml"
        echo "  full      - Full experiment (3 models)"
        exit 1
        ;;
esac

echo ""
echo "✅ Done!"