-- Add strategy column to paper trading tables
-- Run this migration to add support for multiple trading strategies

-- Add strategy column to positions table
ALTER TABLE paper_trading_positions
ADD COLUMN IF NOT EXISTS strategy VARCHAR(50) DEFAULT 'default';

-- Add strategy column to orders table
ALTER TABLE paper_trading_orders
ADD COLUMN IF NOT EXISTS strategy VARCHAR(50) DEFAULT 'default';

-- Add strategy column to account table
ALTER TABLE paper_trading_account
ADD COLUMN IF NOT EXISTS strategy VARCHAR(50) DEFAULT 'default';

-- Create indexes for faster queries by strategy
CREATE INDEX IF NOT EXISTS idx_positions_strategy ON paper_trading_positions(strategy);
CREATE INDEX IF NOT EXISTS idx_orders_strategy ON paper_trading_orders(strategy);
CREATE INDEX IF NOT EXISTS idx_account_strategy ON paper_trading_account(strategy);
