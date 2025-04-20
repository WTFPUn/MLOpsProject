import unittest
import boto3
from botocore.exceptions import NoCredentialsError, ClientError


class TestS3Client(unittest.TestCase):
    
    def setUp(self):
        self.s3 = boto3.client('s3', region_name='ap-southeast-1')  # specify your region
        self.bucket_name = 'kmuttcpe393datamodelnewssum'
        self.file_key = "test/test.txt"  # S3 file key
        self.local_file = 'test.txt'  # Local file to upload

    def test_read_from_s3(self):
        """Test reading a file from S3"""
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=self.file_key)
            content = response['Body'].read().decode('utf-8')
            self.assertIsNotNone(content, "File content should not be None")
        except NoCredentialsError:
            self.fail("Credentials not available")
        except ClientError as e:
            self.fail(f"Failed to read from S3: {e}")

    def test_write_to_s3(self):
        """Test writing a file to S3"""
        try:
            with open(self.local_file, 'w') as f:
                f.write('This is a test content.')

            self.s3.upload_file(self.local_file, self.bucket_name, self.file_key)
            response = self.s3.get_object(Bucket=self.bucket_name, Key=self.file_key)
            content = response['Body'].read().decode('utf-8')

            self.assertEqual(content, 'This is a test content.', "The content uploaded is not correct.")
        except NoCredentialsError:
            self.fail("Credentials not available")
        except ClientError as e:
            self.fail(f"Failed to write to S3: {e}")

    def test_update_s3_file(self):
        """Test updating a file on S3"""
        updated_content = 'This is updated content.'
        
        try:
            # Write initial content to the file
            with open(self.local_file, 'w') as f:
                f.write(updated_content)

            # Upload updated content
            self.s3.upload_file(self.local_file, self.bucket_name, self.file_key)
            
            # Read back the content
            response = self.s3.get_object(Bucket=self.bucket_name, Key=self.file_key)
            content = response['Body'].read().decode('utf-8')
            self.assertEqual(content, updated_content, "The content after update is incorrect.")
        except NoCredentialsError:
            self.fail("Credentials not available")
        except ClientError as e:
            self.fail(f"Failed to update file on S3: {e}")


if __name__ == '__main__':
    unittest.main()
