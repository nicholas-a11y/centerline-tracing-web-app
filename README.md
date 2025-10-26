# Centerline Extraction Web App

A web-based tool for interactive centerline extraction from bitmap images using circle-based evaluation.

## Features
- Interactive parameter adjustment
- Real-time processing
- SVG output generation
- Web-based interface

## Deployment

### Heroku
1. Install Heroku CLI
2. `heroku create your-app-name`
3. `git init && git add . && git commit -m "Initial commit"`
4. `heroku git:remote -a your-app-name`
5. `git push heroku main`

### Railway
1. Connect GitHub repository
2. Deploy automatically

### Render
1. Connect GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `python centerline_web_app.py`

## Local Development
```bash
pip install -r requirements.txt
python centerline_web_app.py
```
