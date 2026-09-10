"""
CLI Script to manually seed demo repositories into CodeLearn AI.
Usage: python seed_repos.py
"""
import logging
import sys

from seed_service import run_demo_seed_pipeline, DEMO_REPOSITORIES
from config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if __name__ == "__main__":
    print("==========================================================")
    print("      CodeLearn AI — Demo Repository Seeder Utility       ")
    print("==========================================================")
    print(f"Target Repositories ({len(DEMO_REPOSITORIES)}):")
    for r in DEMO_REPOSITORIES:
        print(f" - {r['url']} ({r['branch']})")
    print("==========================================================")
    
    # Temporarily force enable for CLI manual invocation
    settings = get_settings()
    settings.SEED_DEMO_REPOS = True
    
    print("Executing ingestion pipeline synchronously...")
    run_demo_seed_pipeline(background=False)
    print("Seeding execution complete.")
