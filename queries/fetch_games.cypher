MATCH (p:Player {name: $playerId}) 
MATCH (g: Game)-[r1:WHITE_PLAYER|BLACK_PLAYER]->(p)
MATCH (g)-[r2:OPENING]-(o:Opening)
return p,g,o,r1,r2
LIMIT 200