/* 
    Template for database setup. Requires strings to be escaped:
    - &dbName
    - &schema
    - &owner
    - &user
    - &password
    - etc.

    TODO: just here for orientation; needs massive restructuring to actually work.
    Probably has tons of bugs as well; need to try out...

    2019 Benjamin Kellenberger
*/


/* schema */
CREATE SCHEMA IF NOT EXISTS &schema
    AUTHORIZATION &user;


/* tables */
CREATE TABLE IF NOT EXISTS &schema.USER (
    name VARCHAR UNIQUE NOT NULL,
    email VARCHAR,
    hash BYTEA,
    isAdmin BOOLEAN DEFAULT FALSE,
    session_token VARCHAR,
    last_login TIMESTAMPTZ,
    PRIMARY KEY (name)
);

CREATE TABLE IF NOT EXISTS &schema.IMAGE (
    id uuid DEFAULT uuid_generate_v4(),
    filename VARCHAR NOT NULL,
    exif VARCHAR,
    fVec bytea,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS &schema.IMAGE_USER (
    username VARCHAR NOT NULL,
    image uuid NOT NULL,
    viewcount SMALLINT DEFAULT 1,

    PRIMARY KEY (username, image),
    FOREIGN KEY (username) REFERENCES &schema.USER(name),
    FOREIGN KEY (image) REFERENCES &schema.IMAGE(id)
);

CREATE TABLE IF NOT EXISTS &schema.LABELCLASS (
    id uuid DEFAULT uuid_generate_v4(),
    name VARCHAR NOT NULL,
    color VARCHAR,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS &schema.ANNOTATION (
    id uuid DEFAULT uuid_generate_v4(),
    username VARCHAR NOT NULL,
    image uuid NOT NULL,
    timeCreated TIMESTAMP NOT NULL DEFAULT NOW(),
    timeRequired BIGINT,
    
    &annotationFields
    PRIMARY KEY (id),
    FOREIGN KEY (username) REFERENCES &schema.USER(name),
    FOREIGN KEY (image) REFERENCES &schema.IMAGE(id)
);

CREATE TABLE IF NOT EXISTS &schema.PREDICTION (
    id uuid DEFAULT uuid_generate_v4(),
    image uuid NOT NULL,
    timeCreated TIMESTAMP NOT NULL DEFAULT NOW(),
    &predictionFields
    priority real,
    PRIMARY KEY (id),
    FOREIGN KEY (image) REFERENCES &schema.IMAGE(id)
);

CREATE TABLE IF NOT EXISTS &schema.CNN (
    id uuid DEFAULT uuid_generate_v4(),
    name VARCHAR NOT NULL,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS &schema.CNN_LABELCLASS (
    cnn uuid NOT NULL,
    labelclass uuid NOT NULL,
    labelNumber BIGINT NOT NULL,
    PRIMARY KEY (cnn, labelclass),
    FOREIGN KEY (cnn) REFERENCES &schema.CNN(id),
    FOREIGN KEY (labelclass) REFERENCES &schema.LABELCLASS(id)
);

CREATE TABLE IF NOT EXISTS &schema.CNNSTATE (
    id uuid DEFAULT uuid_generate_v4(),
    cnn uuid NOT NULL,
    timeCreated TIMESTAMP NOT NULL DEFAULT NOW(),
    stateDict bytea NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY (cnn) REFERENCES &schema.CNN(id)
);

/* TODO: integrate user account tables, reference from annotation table */

