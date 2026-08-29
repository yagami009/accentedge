#!/usr/bin/env python3
"""
Accent Voice API server runner.

Usage:
    python scripts/run_api.py
    HOST=0.0.0.0 PORT=8000 python scripts/run_api.py
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import uvicorn

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the server."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Quiet down noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Accent Voice API server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", "0.0.0.0"),
        help="Server host address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 8000)),
        help="Server port",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development (not recommended for production)",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "info"),
        choices=["critical", "error", "warning", "info", "debug"],
        help="Logging level",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("WORKERS", 1)),
        help="Number of worker processes (use 1 for model serving)",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point for the API server."""
    args = parse_args()
    setup_logging(verbose=(args.log_level == "debug"))

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Starting Accent Voice API")
    logger.info(f"  Host:    {args.host}")
    logger.info(f"  Port:    {args.port}")
    logger.info(f"  Workers: {args.workers}")
    logger.info(f"  LogLevel: {args.log_level}")
    logger.info("=" * 60)

    # Warn if using multiple workers (model loading will happen per worker)
    if args.workers > 1:
        logger.warning(
            "Multiple workers requested — each worker will load its own copy "
            "of the model. This significantly increases memory usage."
        )

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        workers=args.workers,
        timeout_keep_alive=120,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
