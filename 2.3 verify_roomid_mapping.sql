WITH queried_loc AS (
	SELECT ST_Transform(ST_SetSRID(ST_MakePoint("longitude", "latitude"),3857), 4326) AS pt_geom, "SpaceID"
	FROM public.location
	WHERE "BuildingID" = 2 AND "Floor" = 3
)

SELECT rg."gid", array_agg(DISTINCT ql."SpaceID")
FROM public.roomgeoms AS rg, queried_loc AS ql 
WHERE ST_Contains(rg."geom", ql."pt_geom")
GROUP BY rg."gid";

