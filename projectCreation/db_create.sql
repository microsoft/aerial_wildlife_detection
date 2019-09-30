/* 
    Template for database setup, to be used by the "projectCreation/setupDB.py" script only.

    2019 Benjamin Kellenberger
*/


/* administrative schema */
CREATE SCHEMA IF NOT EXISTS aide_admin
    AUTHORIZATION &user;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'labeltype') THEN
        create type labelType AS ENUM ('labels', 'points', 'boundingBoxes', 'segmentationMasks');
    END IF;
END
$$;

CREATE TABLE IF NOT EXISTS aide_admin.project (
    shortname VARCHAR UNIQUE NOT NULL,
    name VARCHAR UNIQUE NOT NULL,
    description VARCHAR,
    isPublic BOOLEAN DEFAULT FALSE,
    secret_token VARCHAR,
    demoMode BOOLEAN DEFAULT FALSE,
    annotationType labelType NOT NULL,
    predictionType labelType,
    ui_settings VARCHAR,
    numImages_autoTrain BIGINT,
    minNumAnnoPerImage INTEGER,
    maxNumImages_train BIGINT,
    maxNumImages_inference BIGINT,
    ai_model_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    ai_model_library VARCHAR,
    ai_model_settings VARCHAR,
    ai_alCriterion_library VARCHAR,
    ai_alCriterion_settings VARCHAR,
    PRIMARY KEY(shortname)
);

CREATE TABLE IF NOT EXISTS aide_admin.user (
    name VARCHAR UNIQUE NOT NULL,
    email VARCHAR,
    hash BYTEA,
    isSuperuser BOOLEAN DEFAULT FALSE,
    session_token VARCHAR,
    last_login TIMESTAMPTZ,
    PRIMARY KEY (name)
);

CREATE TABLE IF NOT EXISTS aide_admin.AUTHENTICATION (
    username VARCHAR UNIQUE NOT NULL,
    project VARCHAR NOT NULL,
    isAdmin BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (username, project),
    FOREIGN KEY (username) REFERENCES aide_admin.user (name),
    FOREIGN KEY (project) REFERENCES aide_admin.project (shortname)
);



/* project schema */
CREATE SCHEMA IF NOT EXISTS &schema
    AUTHORIZATION &user;


/* tables */
/*
CREATE TABLE IF NOT EXISTS &schema.USER (
    name VARCHAR UNIQUE NOT NULL,
    email VARCHAR,
    hash BYTEA,
    isAdmin BOOLEAN DEFAULT FALSE,
    session_token VARCHAR,
    last_login TIMESTAMPTZ,
    PRIMARY KEY (name)
);
*/

CREATE TABLE IF NOT EXISTS &schema.IMAGE (
    id uuid DEFAULT uuid_generate_v4(),
    filename VARCHAR UNIQUE NOT NULL,
    exif VARCHAR,
    fVec bytea,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS &schema.IMAGE_USER (
    username VARCHAR NOT NULL,
    image uuid NOT NULL,
    viewcount SMALLINT DEFAULT 1,
    last_checked TIMESTAMPTZ,
    last_time_required BIGINT,
    meta VARCHAR,

    PRIMARY KEY (username, image),
    FOREIGN KEY (username) REFERENCES aide_admin.user(name),
    FOREIGN KEY (image) REFERENCES &schema.IMAGE(id)
);

CREATE TABLE IF NOT EXISTS &schema.LABELCLASSGROUP (
    id uuid DEFAULT uuid_generate_v4(),
    name VARCHAR NOT NULL,
    color VARCHAR,
    parent uuid,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS &schema.LABELCLASS (
    id uuid DEFAULT uuid_generate_v4(),
    name VARCHAR UNIQUE NOT NULL,
    idx SERIAL UNIQUE NOT NULL,
    color VARCHAR,
    labelclassgroup uuid,
    keystroke SMALLINT UNIQUE,
    PRIMARY KEY (id),
    FOREIGN KEY (labelclassgroup) REFERENCES &schema.LABELCLASSGROUP(id)
);

CREATE TABLE IF NOT EXISTS &schema.ANNOTATION (
    id uuid DEFAULT uuid_generate_v4(),
    username VARCHAR NOT NULL,
    image uuid NOT NULL,
    meta VARCHAR,
    timeCreated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    timeRequired BIGINT,
    unsure boolean NOT NULL DEFAULT false,
    &annotationFields
    PRIMARY KEY (id),
    FOREIGN KEY (username) REFERENCES aide_admin.user(name),
    FOREIGN KEY (image) REFERENCES &schema.IMAGE(id)
);

/*
CREATE TABLE IF NOT EXISTS &schema.CNN (
    id uuid DEFAULT uuid_generate_v4(),
    name VARCHAR NOT NULL,
    PRIMARY KEY (id)
);
*/

/*
CREATE TABLE IF NOT EXISTS &schema.CNN_LABELCLASS (
    cnn uuid NOT NULL,
    labelclass uuid NOT NULL,
    labelNumber BIGINT NOT NULL,
    PRIMARY KEY (cnn, labelclass),
    FOREIGN KEY (cnn) REFERENCES &schema.CNN(id),
    FOREIGN KEY (labelclass) REFERENCES &schema.LABELCLASS(id)
);
*/

CREATE TABLE IF NOT EXISTS &schema.CNNSTATE (
    id uuid DEFAULT uuid_generate_v4(),
    --cnn uuid NOT NULL,
    model_library VARCHAR,
    timeCreated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    stateDict bytea NOT NULL,
    partial boolean NOT NULL,
    PRIMARY KEY (id)
    --, FOREIGN KEY (cnn) REFERENCES &schema.CNN(id)
);

CREATE TABLE IF NOT EXISTS &schema.PREDICTION (
    id uuid DEFAULT uuid_generate_v4(),
    image uuid NOT NULL,
    timeCreated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    cnnstate UUID,
    confidence real,
    &predictionFields
    priority real,
    PRIMARY KEY (id),
    FOREIGN KEY (image) REFERENCES &schema.IMAGE(id),
    FOREIGN KEY (cnnstate) REFERENCES &schema.CNNSTATE(id)
);