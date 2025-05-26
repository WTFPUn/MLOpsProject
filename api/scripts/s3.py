from botocore.exceptions import NoCredentialsError, ClientError
import boto3
import dotenv
dotenv_path = dotenv.find_dotenv()
if dotenv_path:
    print(f"Found .env file at {dotenv_path}")
    # export dotenv_path to environment variables
    dotenv.load_dotenv(dotenv_path)
    
prefix = "news_summary/news_week"
s3 = boto3.client('s3', region_name='ap-southeast-1')  # specify your region
bucket_name = 'kmuttcpe393datamodelnewssum'

def access_s3_file():
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    # Print the matching file keys
    if 'Contents' in response:
        for obj in response['Contents']:
            print(obj['Key'])
    else:
        print("No files found with the specified prefix.")
        
if __name__ == "__main__":
    access_s3_file()
