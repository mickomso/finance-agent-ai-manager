import json
import logging

from finance_agent.agent import FinanceAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Hello from finance-manager!")
    agent = FinanceAgent(model="google/gemma-4-12b-qat")
    logger.info("Created FinanceAgent instance: %s", json.dumps(agent.__dict__))

    result = agent.classify_expense("I bought food at the supermarket for $50")
    logger.info("Classified expense result: %s", json.dumps(result.__dict__))


if __name__ == "__main__":
    main()
