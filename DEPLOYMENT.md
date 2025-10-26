# 🌐 Web Deployment Guide for Centerline Extraction App

## 🚀 Quick Start - Recommended Platforms

### 1. 🌟 **Railway** (Easiest - Recommended for Beginners)

**Pros:** Free tier, automatic deployments, simple setup
**Cons:** Limited free hours

**Steps:**
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway automatically detects Python and deploys!
6. Get your live URL instantly

**Cost:** Free tier with limitations, $5/month for more resources

---

### 2. 🎯 **Render** (Great Balance)

**Pros:** Free tier, easy setup, good performance
**Cons:** Cold starts on free tier

**Steps:**
1. Go to [render.com](https://render.com)
2. Connect your GitHub account
3. Click "New" → "Web Service"
4. Connect repository
5. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python centerline_web_app.py`
6. Deploy!

**Cost:** Free tier available, $7/month for always-on

---

### 3. 🔷 **Heroku** (Popular Choice)

**Pros:** Well-documented, many add-ons
**Cons:** No free tier anymore

**Steps:**
1. Install [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
2. Run these commands:
```bash
# Login to Heroku
heroku login

# Create app
heroku create your-centerline-app

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main

# Open app
heroku open
```

**Cost:** $7/month minimum

---

## 🛠 Setup Your Repository for Deployment

### Step 1: Initialize Git Repository
```bash
cd /Users/nicholas
git init
git add .
git commit -m "Initial commit for centerline extraction app"
```

### Step 2: Push to GitHub
1. Create new repository on [GitHub](https://github.com)
2. Run:
```bash
git remote add origin https://github.com/yourusername/centerline-extraction.git
git branch -M main
git push -u origin main
```

### Step 3: Choose deployment platform and follow steps above

---

## 🔧 Advanced Deployment Options

### 4. **PythonAnywhere** (Python-focused)
- Upload files via web interface
- Create Flask web app
- Configure WSGI file
- **Cost:** Free tier with limitations

### 5. **Google Cloud Platform** (Enterprise)
- Use App Engine with the provided `app.yaml`
- Run: `gcloud app deploy`
- **Cost:** Pay-as-you-go

### 6. **AWS Elastic Beanstalk** (Enterprise)
- Zip your application
- Upload to Elastic Beanstalk
- **Cost:** Pay-as-you-go

---

## 📋 Required Files (Already Created)

✅ `requirements.txt` - Python dependencies
✅ `Procfile` - Process configuration
✅ `app.yaml` - Google Cloud configuration  
✅ `README.md` - Documentation
✅ All source files ready for deployment

---

## 🎯 **My Recommendation:**

**For Quick Testing:** Use **Railway** - literally just connect GitHub and click deploy

**For Serious Use:** Use **Render** - good free tier, then affordable paid plans

**For Enterprise:** Use **Google Cloud** or **AWS**

---

## 🔒 Important Notes

1. **File Size Limits:** Some platforms limit upload sizes. Your image processing might hit limits on free tiers.

2. **Memory Usage:** Image processing uses RAM. Monitor usage and upgrade if needed.

3. **Environment Variables:** Set `FLASK_ENV=production` for production deployments.

4. **HTTPS:** All platforms provide free HTTPS certificates.

5. **Custom Domain:** Most platforms support custom domains on paid plans.

---

## 🚀 Ready to Deploy?

Choose your platform and follow the steps above. Your centerline extraction tool will be live on the web in minutes!

**Need help?** Each platform has excellent documentation and support.
