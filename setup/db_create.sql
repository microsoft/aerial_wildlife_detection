/* 
    Template for database setup, to be used by the "setup/setupDB.py" script only.
    Unlike in previous versions of AIDE, this does not set up any project-specific
    schemata, but only the administrative environment of the Postgres database.
    For project creation, see modules.ProjectAdministration.static.sql.create_schema.sql.

    2019-20 Benjamin Kellenberger
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
    owner VARCHAR,
    isPublic BOOLEAN DEFAULT FALSE,
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
    default_workflow UUID,
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
    canCreateProjects BOOLEAN DEFAULT FALSE,
    session_token VARCHAR,
    secret_token VARCHAR DEFAULT md5(random()::text),
    last_login TIMESTAMPTZ,
    PRIMARY KEY (name)
);

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
