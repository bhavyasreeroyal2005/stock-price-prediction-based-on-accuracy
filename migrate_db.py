#!/usr/bin/env python3
"""
Database migration script to add preferred_algorithm column
"""

import pymysql
from config import Config

def add_column():
    """Add preferred_algorithm column to user table"""
    try:
        # Connect to MySQL database
        connection = pymysql.connect(
            host=Config.MYSQL_HOST,
            port=int(Config.MYSQL_PORT),
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
            database=Config.MYSQL_DATABASE,
            charset='utf8mb4'
        )

        with connection.cursor() as cursor:
            # Check if column exists
            cursor.execute("SHOW COLUMNS FROM user LIKE 'preferred_algorithm'")
            result = cursor.fetchone()

            if result:
                print("✅ Column 'preferred_algorithm' already exists")
            else:
                # Add the column
                cursor.execute('ALTER TABLE user ADD COLUMN preferred_algorithm VARCHAR(20) DEFAULT "LSTM"')
                print("✅ Column 'preferred_algorithm' added successfully")

        connection.commit()
        connection.close()
        return True

    except Exception as e:
        print(f"❌ Error adding column: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Adding preferred_algorithm column to user table...")
    print("=" * 60)

    if add_column():
        print("\n🎉 Migration completed successfully!")
    else:
        print("\n❌ Migration failed!")
