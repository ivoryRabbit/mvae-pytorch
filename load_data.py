from utils import read_csv_s3, read_parquet_s3, download_s3_object


user_meta = read_csv_s3("marketing/user-meta")
item_meta = read_csv_s3("marketing/item-meta")
purchase = read_csv_s3("marketing/bulk_purchase_event")
target_user = read_parquet_s3("marketing_target_user/2021-09-28")

user_meta.to_csv("data/user_meta.csv", index=False)
item_meta.to_csv("data/item_meta.csv", index=False)
purchase.to_csv("data/purchase.csv", index=False)
target_user.to_csv("data/target_user.csv", index=False)

# Download trained model
download_s3_object("models/mvae.pt", "trained/mvae.pt")