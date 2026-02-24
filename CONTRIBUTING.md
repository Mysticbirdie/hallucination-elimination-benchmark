# Contributing to Hallucination Elimination Benchmark

Thank you for your interest in contributing! This benchmark is designed to be a rigorous testbed for hallucination elimination in LLMs.

## How to Contribute

### Adding New Questions
- Questions must be historically accurate for Ancient Rome, 110 CE
- Focus on anachronism detection, character consistency, cultural values
- Provide clear ground truth answers
- Include category classification

### Adding New Model Runners
- Follow the existing pattern in `runners/` directory
- Include proper error handling and resume capability
- Use Gemini 2.0 Flash as independent judge
- Document API requirements

### Reporting Issues
- Provide detailed reproduction steps
- Include model version and context used
- Share expected vs actual results

### Data Quality
- All historical claims must be sourced
- Cultural references should be period-accurate
- Avoid modern concepts that didn't exist in 110 CE

## Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Validate questions
python tools/validate_questions.py
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
