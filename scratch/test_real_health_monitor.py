import asyncio
import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import connector
from modules.connector import get_api
import config
import main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
log = logging.getLogger("test_real_health_monitor")

# Override HEALTH_CHECK_S for faster testing
config.HEALTH_CHECK_S = 3
config.SYMBOLS = ["R_75"]

async def run_test():
    # Initialize connection
    log.info("Initializing API connection...")
    success = await connector.initialize()
    if not success:
        log.error("Failed to connect.")
        return
        
    api = get_api()
    main._init_symbol_state()
    
    # Initially trigger portfolio state sync to match actual startup sequence
    await main.sync_portfolio_state()
    
    # Start the real health monitor from main.py as a background task
    monitor_task = asyncio.create_task(main.health_monitor())
    
    # Wait to ensure healthy connection state
    await asyncio.sleep(2)
    log.info(f"Initial connection healthy: {api.connection_healthy}")
    log.info(f"Is portfolio synced: {main._portfolio_synced}")
    
    # Simulating abrupt disconnect by closing the websocket
    log.info("=== Simulating abrupt disconnect by closing the websocket ===")
    await api._ws.close()
    
    # Wait for health monitor to run, reconnect with backoff, and sync portfolio
    log.info("Waiting 10 seconds for health monitor to reconnect and sync portfolio...")
    await asyncio.sleep(10)
    
    log.info(f"Connection healthy after reconnect: {api.connection_healthy}")
    log.info(f"Is portfolio synced after reconnect: {main._portfolio_synced}")
    
    assert api.connection_healthy is True, "Expected API to be healthy after reconnect"
    assert main._portfolio_synced is True, "Expected portfolio to be synced after reconnect"
    
    # Cleanup
    log.info("Cleaning up...")
    main._shutdown_event.set()
    await monitor_task
    await api.disconnect()
    log.info("Real health monitor reconnection test passed successfully!")

if __name__ == "__main__":
    asyncio.run(run_test())
