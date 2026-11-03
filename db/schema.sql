-- AgroQuery schema. Re-applied from scratch by `agroquery data`.
-- COMMENT ON statements double as the semantic documentation given to the agent.

DROP TABLE IF EXISTS alerts, sensor_readings, sensors, harvests, fields, farms,
    weather_daily, crops, regions CASCADE;

CREATE TABLE regions (
    region_id    integer PRIMARY KEY,
    name         text NOT NULL UNIQUE,
    department   text NOT NULL,
    latitude     double precision NOT NULL,
    longitude    double precision NOT NULL,
    elevation_m  integer NOT NULL
);
COMMENT ON TABLE regions IS 'Agricultural zones of Colombia. Each region has one weather series.';
COMMENT ON COLUMN regions.department IS 'Colombian department (state), e.g. Meta, Tolima, Huila.';

CREATE TABLE crops (
    crop              text PRIMARY KEY,
    common_name_es    text NOT NULL,
    cycle             text NOT NULL CHECK (cycle IN ('annual', 'perennial')),
    moisture_min_pct  double precision NOT NULL,
    moisture_max_pct  double precision NOT NULL
);
COMMENT ON TABLE crops IS 'Reference data per crop. Crop codes: coffee, avocado, plantain, oil_palm, rice, cacao, maize, sugarcane.';
COMMENT ON COLUMN crops.cycle IS 'annual = sown and harvested in cycles; perennial = planted once, harvested repeatedly.';
COMMENT ON COLUMN crops.moisture_min_pct IS 'Lower bound of the optimal volumetric soil moisture range (%). Below it the crop is water-stressed.';
COMMENT ON COLUMN crops.moisture_max_pct IS 'Upper bound of the optimal volumetric soil moisture range (%). Above it the soil is waterlogged.';

CREATE TABLE weather_daily (
    region_id               integer NOT NULL REFERENCES regions,
    date                    date NOT NULL,
    temp_mean_c             double precision NOT NULL,
    temp_min_c              double precision NOT NULL,
    temp_max_c              double precision NOT NULL,
    precipitation_mm        double precision NOT NULL,
    rel_humidity_mean_pct   double precision NOT NULL,
    radiation_mj_m2         double precision NOT NULL,
    et0_mm                  double precision NOT NULL,
    PRIMARY KEY (region_id, date)
);
COMMENT ON TABLE weather_daily IS 'Real daily weather per region (Open-Meteo historical archive), 2023-01-01 to 2025-12-31.';
COMMENT ON COLUMN weather_daily.precipitation_mm IS 'Total rainfall of the day in mm. A rainy day is a day with precipitation_mm >= 1.';
COMMENT ON COLUMN weather_daily.radiation_mj_m2 IS 'Daily shortwave solar radiation in MJ/m2.';
COMMENT ON COLUMN weather_daily.et0_mm IS 'FAO-56 reference evapotranspiration of the day in mm.';

CREATE TABLE farms (
    farm_id       integer PRIMARY KEY,
    name          text NOT NULL UNIQUE,
    region_id     integer NOT NULL REFERENCES regions,
    latitude      double precision NOT NULL,
    longitude     double precision NOT NULL,
    joined_date   date NOT NULL
);
COMMENT ON TABLE farms IS 'Farms monitored by the platform. Farm names are unique, e.g. ''La Esperanza''.';
COMMENT ON COLUMN farms.joined_date IS 'Date the farm joined the monitoring platform.';

CREATE TABLE fields (
    field_id       integer PRIMARY KEY,
    farm_id        integer NOT NULL REFERENCES farms,
    name           text NOT NULL,
    crop           text NOT NULL REFERENCES crops,
    area_ha        double precision NOT NULL,
    planting_date  date NOT NULL,
    irrigated      boolean NOT NULL,
    UNIQUE (farm_id, name)
);
COMMENT ON TABLE fields IS 'Fields (lots, "lotes") inside a farm, each planted with a single crop.';
COMMENT ON COLUMN fields.name IS 'Field name, unique only within its farm (e.g. ''Lote 3'').';
COMMENT ON COLUMN fields.area_ha IS 'Planted area in hectares.';
COMMENT ON COLUMN fields.planting_date IS 'For perennial crops, the date the plantation was established. For annual crops, the first sowing date; later cycles are re-sown right after each harvest.';
COMMENT ON COLUMN fields.irrigated IS 'True if the field has an irrigation system.';

CREATE TABLE sensors (
    sensor_id     integer PRIMARY KEY,
    field_id      integer NOT NULL REFERENCES fields,
    sensor_type   text NOT NULL CHECK (sensor_type IN ('soil_moisture', 'soil_temp', 'ph')),
    installed_at  date NOT NULL
);
COMMENT ON TABLE sensors IS 'IoT soil sensors installed in fields. A field has at most one sensor of each type.';
COMMENT ON COLUMN sensors.sensor_type IS 'soil_moisture (volumetric %, every 2 h), soil_temp (degrees C, every 2 h), ph (soil pH, once a day).';

CREATE TABLE sensor_readings (
    sensor_id  integer NOT NULL REFERENCES sensors,
    ts         timestamp NOT NULL,
    value      double precision NOT NULL,
    PRIMARY KEY (sensor_id, ts)
);
COMMENT ON TABLE sensor_readings IS 'Time series of sensor measurements. Units depend on sensors.sensor_type. Sensors occasionally go offline, leaving gaps.';
COMMENT ON COLUMN sensor_readings.ts IS 'Local Colombia time (UTC-5, no daylight saving).';

CREATE TABLE harvests (
    harvest_id     integer PRIMARY KEY,
    field_id       integer NOT NULL REFERENCES fields,
    harvest_date   date NOT NULL,
    yield_kg       double precision NOT NULL,
    quality_grade  char(1) NOT NULL CHECK (quality_grade IN ('A', 'B', 'C'))
);
COMMENT ON TABLE harvests IS 'Harvest events per field. Yield per hectare = yield_kg / fields.area_ha.';
COMMENT ON COLUMN harvests.yield_kg IS 'Harvested mass in kg (coffee: dry parchment; oil palm: fresh fruit bunches; sugarcane: cane stalks).';
COMMENT ON COLUMN harvests.quality_grade IS 'A = best, C = worst.';

CREATE TABLE alerts (
    alert_id    integer PRIMARY KEY,
    sensor_id   integer NOT NULL REFERENCES sensors,
    ts          timestamp NOT NULL,
    alert_type  text NOT NULL CHECK (alert_type IN
                    ('low_moisture', 'high_moisture', 'high_soil_temp', 'low_ph', 'high_ph')),
    level       text NOT NULL CHECK (level IN ('warning', 'critical')),
    value       double precision NOT NULL
);
COMMENT ON TABLE alerts IS 'Alerts raised when a reading crosses a threshold. Raised at the first out-of-range reading of an episode; the same alert is not repeated for a sensor within 24 hours.';
COMMENT ON COLUMN alerts.alert_type IS 'low_moisture / high_moisture: outside the crop''s optimal moisture range; high_soil_temp: soil above 32 C (critical above 34 C); low_ph: pH < 5.0; high_ph: pH > 7.5.';
COMMENT ON COLUMN alerts.level IS 'warning, or critical when far outside the range: moisture more than 5 points outside the optimal range, soil temperature above 34 C, pH below 4.5 or above 8.0.';
COMMENT ON COLUMN alerts.value IS 'Sensor reading that triggered the alert.';

COMMENT ON SCHEMA public IS 'Glossary: "lot"/"lote" = field. Dry season (temporada seca) in these regions = months December, January, February, June, July, August; the other months are the rainy season. "Yield" means yield_kg / area_ha (kg per hectare) unless stated otherwise. Years in the data: 2023, 2024, 2025.';
