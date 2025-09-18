import json


with open("tiktok_category_list.json", "r") as f:
    data = f.read()
    data = json.loads(data)


# print(type(data["aggregations"]))

ads = []

for i in data["aggregations"]["by_industry"]["buckets"]:
    for main in i["top_ads"]["hits"]["hits"]:
        ads.append({"_id": main["_id"],
                    "category": i["key"],
                    "url": main["_source"]["video_url"],
                    # "video_cover": main["_source"]["video_cover"],
                    "ctr_graph": main["_source"]["ctr_graph"]
                    })
        
with open("tiktok_ads.json", "w") as f:
    json.dump(ads, f, indent=4)