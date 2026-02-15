MATCH (p:Player {pid: $playerId}) 
MATCH (g: Game)-[r1:WHITE_PLAYER|BLACK_PLAYER]->(p)
MATCH (g)-[r]-(x)
return p,g,r1,r,x