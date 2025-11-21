# MySQL Database Setup for Stock Price Predictor

This guide will help you set up MySQL database for the Stock Price Predictor application.

## Prerequisites

1. **MySQL Server**: Make sure MySQL is installed and running on your system
2. **Python Dependencies**: All required packages are installed via requirements.txt

## Step 1: Install MySQL Server

### Windows:
1. Download MySQL Installer from https://dev.mysql.com/downloads/installer/
2. Run the installer and follow the setup wizard
3. Remember the root password you set during installation

### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install mysql-server
sudo mysql_secure_installation
```

### macOS:
```bash
brew install mysql
brew services start mysql
mysql_secure_installation
```

## Step 2: Configure Database Connection

1. **Create Environment File**: Copy `env_template.txt` to `.env` and update the values:
   ```bash
   cp env_template.txt .env
   ```

2. **Edit .env file** with your MySQL credentials:
   ```env
   # Database Configuration
   MYSQL_HOST=localhost
   MYSQL_PORT=3306
   MYSQL_USER=root
   MYSQL_PASSWORD=your_mysql_password
   MYSQL_DATABASE=stock_predictor

   # Flask Configuration
   SECRET_KEY=your-secret-key-change-this-in-production
   FLASK_ENV=development
   ```

## Step 3: Create Database and Tables

Run the database setup script:
```bash
python setup_database.py
```

This script will:
- Create the `stock_predictor` database
- Create all necessary tables (User, PredictionHistory)
- Test the database connection

## Step 4: Run the Application

Start the application:
```bash
python app.py
```

## Troubleshooting

### Common Issues:

1. **Connection Refused**:
   - Make sure MySQL server is running
   - Check if the port (3306) is correct
   - Verify the host address

2. **Access Denied**:
   - Check username and password in .env file
   - Make sure the MySQL user has proper permissions

3. **Database Not Found**:
   - Run the setup script: `python setup_database.py`
   - Check if the database name is correct

4. **Permission Denied**:
   - Grant privileges to your MySQL user:
   ```sql
   GRANT ALL PRIVILEGES ON stock_predictor.* TO 'your_username'@'localhost';
   FLUSH PRIVILEGES;
   ```

### Manual Database Creation:

If the setup script fails, you can manually create the database:

1. **Connect to MySQL**:
   ```bash
   mysql -u root -p
   ```

2. **Create Database**:
   ```sql
   CREATE DATABASE stock_predictor CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   ```

3. **Create User (Optional)**:
   ```sql
   CREATE USER 'stock_user'@'localhost' IDENTIFIED BY 'your_password';
   GRANT ALL PRIVILEGES ON stock_predictor.* TO 'stock_user'@'localhost';
   FLUSH PRIVILEGES;
   ```

## Database Schema

The application creates two main tables:

### Users Table:
- `id`: Primary key
- `username`: Unique username
- `email`: Unique email address
- `password_hash`: Hashed password
- `first_name`, `last_name`: User profile
- `created_at`: Registration timestamp
- `last_login`: Last login timestamp
- `is_active`: Account status

### PredictionHistory Table:
- `id`: Primary key
- `user_id`: Foreign key to Users table
- `symbol`: Stock/Mutual fund symbol
- `asset_type`: 'Stock' or 'Mutual Fund'
- `forecast_days`: Number of forecast days
- `current_price`: Current price/NAV
- `predicted_price`: Predicted price/NAV
- `accuracy`: Prediction accuracy percentage
- `created_at`: Prediction timestamp

## Security Notes

1. **Change Default Passwords**: Update the SECRET_KEY and MySQL password
2. **Environment Variables**: Never commit .env file to version control
3. **Database Permissions**: Use least privilege principle for database users
4. **SSL Connection**: Consider enabling SSL for production deployments

## Production Deployment

For production, consider:
- Using a dedicated MySQL user with limited privileges
- Enabling SSL connections
- Setting up database backups
- Using connection pooling
- Monitoring database performance

