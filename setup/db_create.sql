/* 
    Template for database setup, to be used by the "setup/setupDB.py" script only.
    Unlike in previous versions of AIDE, this does not set up any project-specific
    schemata, but only the administrative environment of the Postgres database.
    For project creation, see modules.ProjectAdministration.static.sql.create_schema.sql.

    2019-21 Benjamin Kellenberger
*/


/* administrative schema */
CREATE SCHEMA IF NOT EXISTS aide_admin
    AUTHORIZATION &user;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'labeltype') THEN
        create type labelType AS ENUM ('labels', 'points', 'boundingBoxes', 'polygons', 'segmentationMasks');
    END IF;
END
$$;

CREATE TABLE IF NOT EXISTS aide_admin.version (
    version VARCHAR UNIQUE NOT NULL,
    PRIMARY KEY (version)
);

CREATE TABLE IF NOT EXISTS aide_admin.project (
    shortname VARCHAR UNIQUE NOT NULL,
    name VARCHAR UNIQUE NOT NULL,
    description VARCHAR,
    owner VARCHAR,
    isPublic BOOLEAN DEFAULT FALSE,
    archived BOOLEAN DEFAULT FALSE,
    secret_token VARCHAR,
    interface_enabled BOOLEAN DEFAULT FALSE,
    demoMode BOOLEAN DEFAULT FALSE,
    annotationType labelType NOT NULL,
    predictionType labelType,
    ui_settings VARCHAR,
    segmentation_ignore_unlabeled BOOLEAN NOT NULL DEFAULT TRUE,
    numImages_autoTrain BIGINT,
    minNumAnnoPerImage INTEGER,
    maxNumImages_train BIGINT,
    maxNumImages_inference BIGINT,
    inference_chunk_size BIGINT,
    max_num_concurrent_tasks INTEGER,
    default_workflow UUID,
    ai_model_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    ai_model_library VARCHAR,
    ai_model_settings VARCHAR,
    ai_alCriterion_library VARCHAR,
    ai_alCriterion_settings VARCHAR,
    watch_folder_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    watch_folder_remove_missing_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    labelclass_autoupdate BOOLEAN NOT NULL DEFAULT FALSE,
    band_config VARCHAR,
    render_config VARCHAR,
    PRIMARY KEY(shortname)
);

CREATE TABLE IF NOT EXISTS aide_admin.user (
    name VARCHAR UNIQUE NOT NULL,
    email VARCHAR,
    hash BYTEA,
    isSuperuser BOOLEAN DEFAULT FALSE,
    canCreateProjects BOOLEAN DEFAULT FALSE,
    session_token VARCHAR,
    secret_token VARCHAR DEFAULT md5(random()::text),
    last_login TIMESTAMPTZ,
    PRIMARY KEY (name)
);

ALTER TABLE aide_admin.project DROP CONSTRAINT IF EXISTS project_user_fkey;
ALTER TABLE aide_admin.project ADD CONSTRAINT project_user_fkey FOREIGN KEY (owner) REFERENCES aide_admin.USER (name);

CREATE TABLE IF NOT EXISTS aide_admin.authentication (
    username VARCHAR NOT NULL,
    project VARCHAR NOT NULL,
    isAdmin BOOLEAN DEFAULT FALSE,
    admitted_until TIMESTAMPTZ,
    blocked_until TIMESTAMPTZ,
    PRIMARY KEY (username, project),
    FOREIGN KEY (username) REFERENCES aide_admin.user (name),
    FOREIGN KEY (project) REFERENCES aide_admin.project (shortname)
);


-- IoU function for statistical evaluations
CREATE OR REPLACE FUNCTION "intersection_over_union" (
	"ax" real, "ay" real, "awidth" real, "aheight" real,
	"bx" real, "by" real, "bwidth" real, "bheight" real)
RETURNS real AS $iou$
	DECLARE
		iou real;
	BEGIN
		SELECT (
			CASE WHEN aright < bleft OR bright < aleft OR
				atop < bbottom OR btop < abottom THEN 0.0
			ELSE GREATEST(inters / (unionplus - inters), 0.0)
			END
		) INTO iou
		FROM (
			SELECT 
				((iright - ileft) * (itop - ibottom)) AS inters,
				aarea + barea AS unionplus,
				aleft, aright, atop, abottom,
				bleft, bright, btop, bbottom
			FROM (
				SELECT
					((aright - aleft) * (atop - abottom)) AS aarea,
					((bright - bleft) * (btop - bbottom)) AS barea,
					GREATEST(aleft, bleft) AS ileft,
					LEAST(atop, btop) AS itop,
					LEAST(aright, bright) AS iright,
					GREATEST(abottom, bbottom) AS ibottom,
					aleft, aright, atop, abottom,
					bleft, bright, btop, bbottom
				FROM (
					SELECT (ax - awidth/2) AS aleft, (ay + aheight/2) AS atop,
						(ax + awidth/2) AS aright, (ay - aheight/2) AS abottom,
						(bx - bwidth/2) AS bleft, (by + bheight/2) AS btop,
						(bx + bwidth/2) AS bright, (by - bheight/2) AS bbottom
				) AS qq
			) AS qq2
		) AS qq3;
		RETURN iou;
	END;
$iou$ LANGUAGE plpgsql;

/*
    Last occurrence of substring. Function obtained from here:
    https://wiki.postgresql.org/wiki/Strposrev
*/
CREATE OR REPLACE FUNCTION strposrev(instring text, insubstring text)
RETURNS integer AS
$BODY$
DECLARE result INTEGER;
BEGIN
    IF strpos(instring, insubstring) = 0 THEN
    -- no match
    result:=0;
    ELSEIF length(insubstring)=1 THEN
    -- add one to get the correct position from the left.
    result:= 1 + length(instring) - strpos(reverse(instring), insubstring);
    ELSE 
    -- add two minus the legth of the search string
    result:= 2 + length(instring)- length(insubstring) - strpos(reverse(instring), reverse(insubstring));
    END IF;
    RETURN result;
END;
$BODY$
LANGUAGE plpgsql IMMUTABLE STRICT
COST 4;


-- Model marketplace
CREATE TABLE IF NOT EXISTS aide_admin.modelMarketplace (
    id UUID DEFAULT uuid_generate_v4(),
    name VARCHAR UNIQUE NOT NULL,
    description VARCHAR NOT NULL,
    labelclasses VARCHAR NOT NULL,
    author VARCHAR NOT NULL,
    model_library VARCHAR NOT NULL,
    model_settings VARCHAR,
    annotationType labelType NOT NULL,
    predictionType labelType NOT NULL,
    citation_info VARCHAR,
    license VARCHAR,
    statedict BYTEA,
    timeCreated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    alCriterion_library VARCHAR,
    origin_project VARCHAR,
    origin_uuid UUID,
    origin_uri VARCHAR,
    public BOOLEAN NOT NULL DEFAULT TRUE,
    anonymous BOOLEAN NOT NULL DEFAULT FALSE,
    selectCount INTEGER NOT NULL DEFAULT 0,
    shared BOOLEAN NOT NULL DEFAULT TRUE,
    tags VARCHAR,
    PRIMARY KEY (id)
);