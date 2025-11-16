#!/usr/bin/env python3
"""
Database setup script for MySQL
Run this script to create the database and tables
"""

import pymysql
from config import Config
import os

def create_database():
    """Create MySQL database if it doesn't exist"""
    try:
        # Connect to MySQL server (without specifying database)
        connection = pymysql.connect(
            host=Config.MYSQL_HOST,
            port=int(Config.MYSQL_PORT),
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
            charset='utf8mb4'
        )
        
        with connection.cursor() as cursor:
            # Create database if it doesn't exist
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {Config.MYSQL_DATABASE} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            print(f"✅ Database '{Config.MYSQL_DATABASE}' created or already exists")
            
        connection.close()
        return True
        
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        return False

def create_tables():
    """Create database tables using Flask-SQLAlchemy"""
    try:
        from app import app, db
        
        with app.app_context():
            db.create_all()
            print("✅ Database tables created successfully!")
            return True
            
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False

def test_connection():
    """Test database connection"""
    try:
        from app import app, db
        
        with app.app_context():
            # Test connection by querying a simple table
            result = db.session.execute("SELECT 1")
            print("✅ Database connection successful!")
            return True
            
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Setting up MySQL database for Stock Price Predictor...")
    print("=" * 60)
    
    # Step 1: Create database
    print("Step 1: Creating database...")
    if not create_database():
        print("❌ Failed to create database. Please check your MySQL configuration.")
        exit(1)
    
    # Step 2: Create tables
    print("\nStep 2: Creating tables...")
    if not create_tables():
        print("❌ Failed to create tables. Please check your database connection.")
        exit(1)
    
    # Step 3: Test connection
    print("\nStep 3: Testing connection...")
    if not test_connection():
        print("❌ Database connection test failed.")
        exit(1)
    
    print("\n🎉 Database setup completed successfully!")
    print("You can now run the application with: python app.py")

