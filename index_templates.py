podcast_body = {
                "settings": {"number_of_shards": 3,
                            "refresh_interval": '30s',
                            "index": {"knn": False}},
                "mappings": {
                    "properties": {
                        "title": {"type": "text", "index": "true"},

                        "id": {"type": "keyword", "index": "false"},

                        "episode_num": {"type": "short", "index": "false"},

                        "episode_url": {"type": "keyword", "index": "false"},

                        "mp3_url": {"type": "keyword", "index": "false"},
                        
                        "summary" : {"type": "text", "index": "true"},

                        "content": {"type": "text", "index": "true"}
                                }
                            }
                }

youtube_body = {
                "settings": {"number_of_shards": 3,
                            "refresh_interval": '30s',
                            "index": {"knn": False}},
                "mappings": {
                    "properties": {
                        "title": {"type": "text", "index": "true"},

                        "group_id": {"type": "short", "index": "false"},

                        "video_id": {"type": "keyword", "index": "false"},

                        "playlist_id": {"type": "keyword", "index": "false"},

                        "episode_url": {"type": "keyword", "index": "false"},
                        
                        "description" : {"type": "text", "index": "true"},

                        "length" : {"type": "long", "index": "false"},

                        "publish_date" : {"type": "keyword", "index": "false"},

                        "views" : {"type": "long", "index": "false"},

                        "thumbnail_url" : {"type": "keyword", "index": "false"},

                        "content": {"type": "text", "index": "true"}
                                }
                            }
                }