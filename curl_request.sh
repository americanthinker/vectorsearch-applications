#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <query_term>"
  exit 1
fi
query="$1"
request_body='{
  "_source": [
    "title",
    "doc_id",
    "episode_num"
  ],
  "size": 15,
  "query": {
    "bool": {
      "must": [
        {
          "match": {
            "title": {
              "query": "'"${query}"'"
            }
          }
        }
      ]
    }
  }
}'
curl -k -u admin:admin -X POST 'https://localhost:9200/impact-theory-minilm-196/_search?pretty' -H 'Content-Type: application/json' -d "${request_body}"