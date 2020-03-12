'''
    Provides SQL formulas for e.g. the evaluation of
    annotations directly in postgres.

    2019-20 Benjamin Kellenberger
'''

from enum import Enum


class StatisticalFormulas_user(Enum):
    labels = '''
        {sql_global_start}
        SELECT q1.image AS image, q1id, q2id, q1label, q2label, q1label=q2label AS label_correct
        FROM (
            SELECT image, id AS q1id, label AS q1label
            FROM {id_anno}
            WHERE username = %s
        ) AS q1
        JOIN
        (
            SELECT image, id AS q2id, label AS q2label
            FROM {id_anno}
            WHERE username = %s
        ) AS q2
        ON q1.image = q2.image
        {sql_goldenQuestion}
        ORDER BY image
        {sql_global_end}
    '''

    
    points = '''WITH masterQuery AS (
        SELECT q1.image, q2.username, q1.aid AS id1, q2.aid AS id2, q1.label AS label1, q2.label AS label2,
            |/((q1.x - q2.x)^2 + (q1.y - q2.y)^2) AS euclidean_distance
        FROM (
            SELECT iu.image, iu.username, anno.id AS aid, label, x, y, width, height FROM {id_iu} AS iu
            LEFT OUTER JOIN {id_anno} AS anno
            ON iu.image = anno.image AND iu.username = anno.username
            WHERE iu.username = %s
        ) AS q1
        JOIN (
            SELECT iu.image, iu.username, anno.id AS aid, label, x, y, width, height FROM {id_iu} AS iu
            LEFT OUTER JOIN {id_anno} AS anno
            ON iu.image = anno.image AND iu.username = anno.username
            WHERE iu.username IN %s
        ) AS q2
        ON q1.image = q2.image
        {sql_goldenQuestion}
    ),
    imgStats AS (
        SELECT image, username, COUNT(DISTINCT id2) AS num_pred, COUNT(DISTINCT id1) AS num_target
        FROM masterQuery 
        GROUP BY image, username
    ),
    tp AS (
        SELECT username, image,
        (CASE WHEN mindist > %s OR label1 != label2 THEN NULL ELSE bestMatch.id1 END) AS id1,
        aux.id2, mindist AS dist
        FROM (
            SELECT username, id1, MIN(euclidean_distance) AS mindist
            FROM masterQuery
            GROUP BY username, id1
        ) AS bestMatch
        JOIN (
            SELECT image, id1, id2, label1, label2, euclidean_distance
            FROM masterQuery
            WHERE euclidean_distance <= %s
        ) AS aux
        ON bestMatch.id1 = aux.id1
        AND mindist = euclidean_distance
    )
    SELECT *
    FROM (
        SELECT image, username, num_pred, num_target, min_dist, avg_dist, max_dist,
            tp, GREATEST(fp, num_pred - tp) AS fp, GREATEST(fn, num_target - tp) AS fn
        FROM (
            SELECT image, username, num_pred, num_target, min_dist, avg_dist, max_dist,
                LEAST(tp, num_pred) AS tp, fp, fn
            FROM imgStats
            JOIN (
                SELECT username AS username_, image AS image_,
                    MIN(dist) AS min_dist, AVG(dist) AS avg_dist, MAX(dist) AS max_dist,
                    SUM(tp) AS tp, SUM(fp) AS fp, SUM(fn) AS fn
                FROM (
                    SELECT username, image,
                            dist,
                            (CASE WHEN id1 IS NOT NULL AND id2 IS NOT NULL THEN 1 ELSE 0 END) AS tp,
                            (CASE WHEN id2 IS NULL THEN 1 ELSE 0 END) AS fp,
                            (CASE WHEN id1 IS NULL THEN 1 ELSE 0 END) AS fn
                    FROM (
                        SELECT * FROM tp
                        UNION ALL (
                            SELECT username, image, NULL AS id1, id2, NULL AS dist
                            FROM masterQuery
                            WHERE id1 NOT IN (
                                SELECT id1 FROM tp
                            )
                            AND masterQuery.id2 IS NOT NULL
                        )
                        UNION ALL (
                            SELECT username, image, id1, NULL AS id2, NULL AS dist
                            FROM masterQuery
                            WHERE id2 IS NULL
                            AND id1 IS NOT NULL
                        )
                    ) AS q1
                ) AS q2
                GROUP BY image_, username_
            ) AS statsQuery
            ON imgstats.image = statsQuery.image_
            AND imgstats.username = statsQuery.username_
        ) AS q3
    ) AS q4
    UNION ALL (
        SELECT image, username, num_pred, num_target, NULL AS min_dist, NULL AS avg_dist, NULL AS max_dist, 0 AS tp, num_pred AS fp, 0 AS fn
        FROM imgStats
        WHERE num_target = 0
    )'''


    boundingBoxes = '''WITH masterQuery AS (
        SELECT q1.image, q2.username, q1.aid AS id1, q2.aid AS id2, q1.label AS label1, q2.label AS label2,
            intersection_over_union(q1.x, q1.y, q1.width, q1.height,
                                    q2.x, q2.y, q2.width, q2.height) AS iou
        FROM (
            SELECT iu.image, iu.username, anno.id AS aid, label, x, y, width, height FROM {id_iu} AS iu
            LEFT OUTER JOIN {id_anno} AS anno
            ON iu.image = anno.image AND iu.username = anno.username
            WHERE iu.username = %s
        ) AS q1
        JOIN (
            SELECT iu.image, iu.username, anno.id AS aid, label, x, y, width, height FROM {id_iu} AS iu
            LEFT OUTER JOIN {id_anno} AS anno
            ON iu.image = anno.image AND iu.username = anno.username
            WHERE iu.username IN %s
        ) AS q2
        ON q1.image = q2.image
        {sql_goldenQuestion}
    ),
    imgStats AS (
        SELECT image, username, COUNT(DISTINCT id2) AS num_pred, COUNT(DISTINCT id1) AS num_target
        FROM masterQuery 
        GROUP BY image, username
    ),
    tp AS (
        SELECT username, image,
        (CASE WHEN maxiou < %s OR label1 != label2 THEN NULL ELSE bestMatch.id1 END) AS id1,
        aux.id2, maxiou AS iou
        FROM (
            SELECT username, id1, MAX(iou) AS maxiou
            FROM masterQuery
            GROUP BY username, id1
        ) AS bestMatch
        JOIN (
            SELECT image, id1, id2, label1, label2, iou
            FROM masterQuery
            WHERE iou > 0
        ) AS aux
        ON bestMatch.id1 = aux.id1
        AND maxiou = iou
    )
    SELECT *
    FROM (
        SELECT image, username, num_pred, num_target, min_iou, avg_iou, max_iou,
            tp, GREATEST(fp, num_pred - tp) AS fp, GREATEST(fn, num_target - tp) AS fn
        FROM (
            SELECT image, username, num_pred, num_target, min_iou, avg_iou, max_iou,
                LEAST(tp, num_pred) AS tp, fp, fn
            FROM imgStats
            JOIN (
                SELECT username AS username_, image AS image_,
                    MIN(iou) AS min_iou, AVG(iou) AS avg_iou, MAX(iou) AS max_iou,
                    SUM(tp) AS tp, SUM(fp) AS fp, SUM(fn) AS fn
                FROM (
                    SELECT username, image,
                            iou,
                            (CASE WHEN id1 IS NOT NULL AND id2 IS NOT NULL THEN 1 ELSE 0 END) AS tp,
                            (CASE WHEN id2 IS NULL THEN 1 ELSE 0 END) AS fp,
                            (CASE WHEN id1 IS NULL THEN 1 ELSE 0 END) AS fn
                    FROM (
                        SELECT * FROM tp
                        UNION ALL (
                            SELECT username, image, NULL AS id1, id2, NULL::real AS iou
                            FROM masterQuery
                            WHERE id1 NOT IN (
                                SELECT id1 FROM tp
                            )
                            AND masterQuery.id2 IS NOT NULL
                        )
                        UNION ALL (
                            SELECT username, image, id1, NULL AS id2, NULL::real AS iou
                            FROM masterQuery
                            WHERE id2 IS NULL
                            AND id1 IS NOT NULL
                        )
                    ) AS q1
                ) AS q2
                GROUP BY image_, username_
            ) AS statsQuery
            ON imgstats.image = statsQuery.image_
            AND imgstats.username = statsQuery.username_
        ) AS q3
    ) AS q4
    UNION ALL (
        SELECT image, username, num_pred, num_target, NULL AS min_iou, NULL AS avg_iou, NULL AS max_iou, 0 AS tp, num_pred AS fp, 0 AS fn
        FROM imgStats
        WHERE num_target = 0
    )'''


    segmentationMasks = '''
        {sql_global_start}
        SELECT q1.image AS image, q1id, q1segMask, q2id, q2segMask FROM (
            SELECT image, segmentationMask AS q1segMask FROM {id_anno}
            WHERE username = %s
        ) AS q1
        JOIN (
            SELECT image, segmentationMask AS q2segMask FROM {id_anno}
            WHERE username = %s
        ) AS q2
        ON q1.image = q2.image
        {sql_goldenQuestion}
        {sql_global_end}
    '''



#TODO: rework formulas below
class StatisticalFormulas_model(Enum):
    labels = '''
        {sql_global_start}
        SELECT q1.image AS image, q1id, q2id, q1label, q2label, q1label=q2label AS label_correct
        FROM (
            SELECT image, id AS q1id, label AS q1label
            FROM {id_anno}
            WHERE username = %s
        ) AS q1
        JOIN
        (
            SELECT image, id AS q2id, label AS q2label
            FROM {id_pred}
            WHERE cnnstate = %s
        ) AS q2
        ON q1.image = q2.image
        {sql_goldenQuestion}
        ORDER BY image
        {sql_global_end}
    '''

    
    points = '''WITH masterQuery AS (
        SELECT q1.image, q2.cnnstate, q1.aid AS id1, q2.aid AS id2, q1.label AS label1, q2.label AS label2,
            |/((q1.x - q2.x)^2 + (q1.y - q2.y)^2) AS euclidean_distance
        FROM (
            SELECT iu.image, iu.username, anno.id AS aid, label, x, y, width, height FROM {id_iu} AS iu
            LEFT OUTER JOIN {id_anno} AS anno
            ON iu.image = anno.image AND iu.username = anno.username
            WHERE iu.username = %s
        ) AS q1
        JOIN (
            SELECT pred.image, pred.cnnstate, pred.id AS aid, label, x, y, width, height FROM {id_pred} AS pred
            WHERE pred.cnnstate IN %s
        ) AS q2
        ON q1.image = q2.image
        {sql_goldenQuestion}
    ),
    imgStats AS (
        SELECT image, username, COUNT(DISTINCT id2) AS num_pred, COUNT(DISTINCT id1) AS num_target
        FROM masterQuery 
        GROUP BY image, username
    ),
    tp AS (
        SELECT username, image,
        (CASE WHEN mindist > %s OR label1 != label2 THEN NULL ELSE bestMatch.id1 END) AS id1,
        aux.id2, mindist AS dist
        FROM (
            SELECT username, id1, MIN(euclidean_distance) AS mindist
            FROM masterQuery
            GROUP BY username, id1
        ) AS bestMatch
        JOIN (
            SELECT image, id1, id2, label1, label2, euclidean_distance
            FROM masterQuery
            WHERE euclidean_distance <= %s
        ) AS aux
        ON bestMatch.id1 = aux.id1
        AND mindist = euclidean_distance
    )
    SELECT *
    FROM (
        SELECT image, username, num_pred, num_target, min_dist, avg_dist, max_dist,
            tp, GREATEST(fp, num_pred - tp) AS fp, GREATEST(fn, num_target - tp) AS fn
        FROM (
            SELECT image, username, num_pred, num_target, min_dist, avg_dist, max_dist,
                LEAST(tp, num_pred) AS tp, fp, fn
            FROM imgStats
            JOIN (
                SELECT username AS username_, image AS image_,
                    MIN(dist) AS min_dist, AVG(dist) AS avg_dist, MAX(dist) AS max_dist,
                    SUM(tp) AS tp, SUM(fp) AS fp, SUM(fn) AS fn
                FROM (
                    SELECT username, image,
                            dist,
                            (CASE WHEN id1 IS NOT NULL AND id2 IS NOT NULL THEN 1 ELSE 0 END) AS tp,
                            (CASE WHEN id2 IS NULL THEN 1 ELSE 0 END) AS fp,
                            (CASE WHEN id1 IS NULL THEN 1 ELSE 0 END) AS fn
                    FROM (
                        SELECT * FROM tp
                        UNION ALL (
                            SELECT username, image, NULL AS id1, id2, NULL AS dist
                            FROM masterQuery
                            WHERE id1 NOT IN (
                                SELECT id1 FROM tp
                            )
                            AND masterQuery.id2 IS NOT NULL
                        )
                        UNION ALL (
                            SELECT username, image, id1, NULL AS id2, NULL AS dist
                            FROM masterQuery
                            WHERE id2 IS NULL
                            AND id1 IS NOT NULL
                        )
                    ) AS q1
                ) AS q2
                GROUP BY image_, username_
            ) AS statsQuery
            ON imgstats.image = statsQuery.image_
            AND imgstats.username = statsQuery.username_
        ) AS q3
    ) AS q4
    UNION ALL (
        SELECT image, username, num_pred, num_target, NULL AS min_dist, NULL AS avg_dist, NULL AS max_dist, 0 AS tp, num_pred AS fp, 0 AS fn
        FROM imgStats
        WHERE num_target = 0
    )'''


    boundingBoxes = '''WITH masterQuery AS (
        SELECT q1.image, q2.username, q1.aid AS id1, q2.aid AS id2, q1.label AS label1, q2.label AS label2,
            intersection_over_union(q1.x, q1.y, q1.width, q1.height,
                                    q2.x, q2.y, q2.width, q2.height) AS iou
        FROM (
            SELECT iu.image, iu.username, anno.id AS aid, label, x, y, width, height FROM {id_iu} AS iu
            LEFT OUTER JOIN {id_anno} AS anno
            ON iu.image = anno.image AND iu.username = anno.username
            WHERE iu.username = %s
        ) AS q1
        JOIN (
            SELECT iu.image, iu.username, anno.id AS aid, label, x, y, width, height FROM {id_iu} AS iu
            LEFT OUTER JOIN {id_anno} AS anno
            ON iu.image = anno.image AND iu.username = anno.username
            WHERE iu.username IN %s
        ) AS q2
        ON q1.image = q2.image
        {sql_goldenQuestion}
    ),
    imgStats AS (
        SELECT image, username, COUNT(DISTINCT id2) AS num_pred, COUNT(DISTINCT id1) AS num_target
        FROM masterQuery 
        GROUP BY image, username
    ),
    tp AS (
        SELECT username, image,
        (CASE WHEN maxiou < %s OR label1 != label2 THEN NULL ELSE bestMatch.id1 END) AS id1,
        aux.id2, maxiou AS iou
        FROM (
            SELECT username, id1, MAX(iou) AS maxiou
            FROM masterQuery
            GROUP BY username, id1
        ) AS bestMatch
        JOIN (
            SELECT image, id1, id2, label1, label2, iou
            FROM masterQuery
            WHERE iou > 0
        ) AS aux
        ON bestMatch.id1 = aux.id1
        AND maxiou = iou
    )
    SELECT *
    FROM (
        SELECT image, username, num_pred, num_target, min_iou, avg_iou, max_iou,
            tp, GREATEST(fp, num_pred - tp) AS fp, GREATEST(fn, num_target - tp) AS fn
        FROM (
            SELECT image, username, num_pred, num_target, min_iou, avg_iou, max_iou,
                LEAST(tp, num_pred) AS tp, fp, fn
            FROM imgStats
            JOIN (
                SELECT username AS username_, image AS image_,
                    MIN(iou) AS min_iou, AVG(iou) AS avg_iou, MAX(iou) AS max_iou,
                    SUM(tp) AS tp, SUM(fp) AS fp, SUM(fn) AS fn
                FROM (
                    SELECT username, image,
                            iou,
                            (CASE WHEN id1 IS NOT NULL AND id2 IS NOT NULL THEN 1 ELSE 0 END) AS tp,
                            (CASE WHEN id2 IS NULL THEN 1 ELSE 0 END) AS fp,
                            (CASE WHEN id1 IS NULL THEN 1 ELSE 0 END) AS fn
                    FROM (
                        SELECT * FROM tp
                        UNION ALL (
                            SELECT username, image, NULL AS id1, id2, NULL::real AS iou
                            FROM masterQuery
                            WHERE id1 NOT IN (
                                SELECT id1 FROM tp
                            )
                            AND masterQuery.id2 IS NOT NULL
                        )
                        UNION ALL (
                            SELECT username, image, id1, NULL AS id2, NULL::real AS iou
                            FROM masterQuery
                            WHERE id2 IS NULL
                            AND id1 IS NOT NULL
                        )
                    ) AS q1
                ) AS q2
                GROUP BY image_, username_
            ) AS statsQuery
            ON imgstats.image = statsQuery.image_
            AND imgstats.username = statsQuery.username_
        ) AS q3
    ) AS q4
    UNION ALL (
        SELECT image, username, num_pred, num_target, NULL AS min_iou, NULL AS avg_iou, NULL AS max_iou, 0 AS tp, num_pred AS fp, 0 AS fn
        FROM imgStats
        WHERE num_target = 0
    )'''


    segmentationMasks = '''
        # {sql_global_start}
        # SELECT q1.image AS image, q1id, q1segMask, q2id, q2segMask FROM (
        #     SELECT image, segmentationMask AS q1segMask FROM {id_anno}
        #     WHERE username = %s
        # ) AS q1
        # JOIN (
        #     SELECT image, segmentationMask AS q2segMask FROM {id_anno}
        #     WHERE username = %s
        # ) AS q2
        # ON q1.image = q2.image
        # {sql_goldenQuestion}
        # {sql_global_end}
    '''