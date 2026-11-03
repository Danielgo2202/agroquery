-- Run after bulk loading: secondary indexes, statistics and the read-only role
-- used to execute model-generated SQL.
CREATE INDEX IF NOT EXISTS alerts_sensor_ts_idx ON alerts (sensor_id, ts);
CREATE INDEX IF NOT EXISTS alerts_ts_idx ON alerts (ts);
CREATE INDEX IF NOT EXISTS harvests_field_idx ON harvests (field_id, harvest_date);
CREATE INDEX IF NOT EXISTS fields_farm_idx ON fields (farm_id);
CREATE INDEX IF NOT EXISTS sensors_field_idx ON sensors (field_id);
CREATE INDEX IF NOT EXISTS readings_ts_idx ON sensor_readings (ts);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'agro_reader') THEN
        CREATE ROLE agro_reader LOGIN PASSWORD 'agro_reader';
    END IF;
END $$;
GRANT CONNECT ON DATABASE agroquery TO agro_reader;
GRANT USAGE ON SCHEMA public TO agro_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO agro_reader;
ALTER ROLE agro_reader SET statement_timeout = '30s';
ALTER ROLE agro_reader SET default_transaction_read_only = on;

ANALYZE;
