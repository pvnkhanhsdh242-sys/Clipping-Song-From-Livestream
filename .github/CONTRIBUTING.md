# Contributing

Thanks for improving karaoke-clipper.

## Local Setup
1. Run `make setup`
2. Optional full stack: install `requirements-ml.txt`
3. Create `.env` from `.env.example`

## Development Flow
1. Pick a milestone slice (M1-M7)
2. Add or update tests
3. Run `make test`
4. Run one smoke command for URL or local mode
5. Open PR with evidence in template

## Coding Rules
- Keep local-only mode working without paid APIs
- Keep external lookups isolated behind backend interfaces
- Add clear logs for recoverable failures
- Mark unsupported/partial behavior with explicit TODOs

## Useful Commands
- `make run-url URL=...`
- `make run-file FILE=...`
- `python scripts/build_reference_library.py --input-dir data/reference_songs`
