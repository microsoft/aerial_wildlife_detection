'''
    Provides SQL formulas for e.g. the evaluation of
    annotations directly in postgres.

    2019 Benjamin Kellenberger
'''

from enum import Enum


class StatisticalFormulas(Enum):
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
            SELECT q2img AS image, q1id, q1label, q2id, q2label, euclidean_distance FROM (
                SELECT q1.image AS q1img, q1.id AS q1id, q1.label AS q1label,
                q2.image AS q2img, q2.id AS q2id, q2.label AS q2label,
                |/((q1.x - q2.x)^2 + (q1.y - q2.y)^2) AS euclidean_distance FROM (
                    SELECT iu_a.image, id, label, x, y, width, height
                    FROM {id_anno} AS anno_a
                    RIGHT OUTER JOIN {id_iu} AS iu_a
                    ON anno_a.image = iu_a.image AND anno_a.username = iu_a.username
                    WHERE iu_a.username = %s
                ) AS q1
                JOIN
                (
                    SELECT iu_b.image, id, label, x, y, width, height
                    FROM {id_anno} AS anno_b
                    RIGHT OUTER JOIN {id_iu} AS iu_b
                    ON anno_b.image = iu_b.image AND anno_b.username = iu_b.username
                    WHERE iu_b.username = %s
                ) AS q2
                ON q1.image = q2.image
                {sql_goldenQuestion}
            ) AS q
        )
        {sql_global_start}
        SELECT image, num_pred, num_target, ntp, LEAST(num_pred - ntp, nfp) AS nfp, num_target - ntp AS nfn FROM (
            SELECT image, num_pred, num_target, LEAST(num_target, num_tp) AS ntp, num_fp AS nfp FROM (
                SELECT i_nt AS image, COALESCE(num_pred, 0) AS num_pred, COALESCE(num_target, 0) AS num_target,
                    COALESCE(num_tp, 0) AS num_tp, COALESCE(num_fp, 0) AS num_fp FROM (
                    SELECT image, SUM(is_tp) AS num_tp, SUM(is_fp) AS num_fp
                    FROM (
                        SELECT *,
                            (
                                CASE WHEN q1label = q2label AND (euclidean_distance <= %s) THEN 1
                                ELSE 0
                                END
                            ) AS is_tp,
                            (
                                CASE WHEN q1label <> q2label OR (euclidean_distance > %s) THEN 1
                                ELSE 0
                                END
                            ) AS is_fp
                        FROM masterQuery
                        JOIN (
                            SELECT q2id AS q2id_2, MIN(euclidean_distance) AS minDist
                            FROM masterQuery
                            GROUP BY q2id
                        ) AS bestMatch
                        ON masterQuery.q2id = bestMatch.q2id_2
                    ) AS statsQuery
                    GROUP BY image
                ) AS statsQuery
                LEFT OUTER JOIN (
                    SELECT iu_a.image AS i_np, COUNT(*) AS num_pred
                    FROM {id_anno} AS anno_a
                    RIGHT OUTER JOIN {id_iu} AS iu_a
                    ON anno_a.image = iu_a.image AND anno_a.username = iu_a.username
                    WHERE iu_a.username = %s
                    GROUP BY iu_a.image
                ) AS q_np
                ON statsQuery.image = q_np.i_np
                FULL OUTER JOIN (
                    SELECT iu_b.image AS i_nt, COUNT(*) AS num_target
                    FROM {id_anno} AS anno_b
                    RIGHT OUTER JOIN {id_iu} AS iu_b
                    ON anno_b.image = iu_b.image AND anno_b.username = iu_b.username
                    WHERE iu_b.username = %s
                    GROUP BY iu_b.image
                ) AS q_nt
                ON statsQuery.image = q_nt.i_nt
            ) AS aggregateQuery
        ) AS finalQuery
        {sql_global_end}
    '''

    
    boundingBoxes = '''WITH masterQuery AS (
            SELECT q2img AS image, q1id, q1label, q2id, q2label, iou FROM (
                SELECT q1.image AS q1img, q1.id AS q1id, q1.label AS q1label,
                q2.image AS q2img, q2.id AS q2id, q2.label AS q2label,
                intersection_over_union(q1.x, q1.y, q1.width, q1.height, q2.x, q2.y, q2.width, q2.height) AS iou FROM (
                    SELECT iu_a.image, id, label, x, y, width, height
                    FROM {id_anno} AS anno_a
                    RIGHT OUTER JOIN {id_iu} AS iu_a
                    ON anno_a.image = iu_a.image AND anno_a.username = iu_a.username
                    WHERE iu_a.username = %s
                ) AS q1
                JOIN
                (
                    SELECT iu_b.image, id, label, x, y, width, height
                    FROM {id_anno} AS anno_b
                    RIGHT OUTER JOIN {id_iu} AS iu_b
                    ON anno_b.image = iu_b.image AND anno_b.username = iu_b.username
                    WHERE iu_b.username = %s
                ) AS q2
                ON q1.image = q2.image
                {sql_goldenQuestion}
            ) AS q
        )
        {sql_global_start}
        SELECT image, num_pred, num_target, ntp, LEAST(num_pred - ntp, nfp) AS nfp, num_target - ntp AS nfn FROM (
            SELECT image, num_pred, num_target, LEAST(num_target, num_tp) AS ntp, num_fp AS nfp FROM (
                SELECT i_nt AS image, COALESCE(num_pred, 0) AS num_pred, COALESCE(num_target, 0) AS num_target,
                    COALESCE(num_tp, 0) AS num_tp, COALESCE(num_fp, 0) AS num_fp FROM (
                    SELECT image, SUM(is_tp) AS num_tp, SUM(is_fp) AS num_fp
                    FROM (
                        SELECT *,
                            (
                                CASE WHEN q1label = q2label AND (iou >= %s) THEN 1
                                ELSE 0
                                END
                            ) AS is_tp,
                            (
                                CASE WHEN q1label <> q2label OR (iou < %s) THEN 1
                                ELSE 0
                                END
                            ) AS is_fp
                        FROM masterQuery
                        JOIN (
                            SELECT q2id AS q2id_2, MAX(iou) AS maxIoU
                            FROM masterQuery
                            GROUP BY q2id
                        ) AS bestMatch
                        ON masterQuery.q2id = bestMatch.q2id_2
                    ) AS statsQuery
                    GROUP BY image
                ) AS statsQuery
                LEFT OUTER JOIN (
                    SELECT iu_a.image AS i_np, COUNT(*) AS num_pred
                    FROM {id_anno} AS anno_a
                    RIGHT OUTER JOIN {id_iu} AS iu_a
                    ON anno_a.image = iu_a.image AND anno_a.username = iu_a.username
                    WHERE iu_a.username = %s
                    GROUP BY iu_a.image
                ) AS q_np
                ON statsQuery.image = q_np.i_np
                FULL OUTER JOIN (
                    SELECT iu_b.image AS i_nt, COUNT(*) AS num_target
                    FROM {id_anno} AS anno_b
                    RIGHT OUTER JOIN {id_iu} AS iu_b
                    ON anno_b.image = iu_b.image AND anno_b.username = iu_b.username
                    WHERE iu_b.username = %s
                    GROUP BY iu_b.image
                ) AS q_nt
                ON statsQuery.image = q_nt.i_nt
            ) AS aggregateQuery
        ) AS finalQuery
        {sql_global_end}
    '''

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