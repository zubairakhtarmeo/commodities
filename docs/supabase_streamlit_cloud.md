# Supabase + Streamlit Cloud (Long-Term Sharing)

This dashboard can be hosted on Streamlit Cloud and use Supabase to store predictions so management can access a stable URL any time.

## 1) Create Supabase project
- Create a Supabase project
- Go to **Project Settings → API**
- Copy:
  - `SUPABASE_URL`
  - `SUPABASE_SERVICE_ROLE_KEY` (recommended for server-side writes)

## 2) Create the table
Run this in Supabase **SQL Editor**:

```sql
create table if not exists prediction_records (
  id bigserial primary key,
  asset text not null,
  as_of_date date not null,
  target_date date not null,
  predicted_value numeric not null,
  actual_value numeric null,
  unit text null,
  model_name text not null default 'default',
  frequency text not null default 'daily',
  horizon text not null default '1d',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create unique index if not exists prediction_records_unique
on prediction_records(asset, as_of_date, target_date, model_name, horizon);
```

## 3) Add Streamlit Cloud Secrets
In Streamlit Cloud:
- App → **Settings** → **Secrets**

Add:

```toml
SUPABASE_URL = "https://YOURPROJECT.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "YOUR_SERVICE_ROLE_KEY"
```

## 4) Use the AI Predictions page
- Open the app
- Go to **🤖 AI Predictions**
- It will show **Actual (bars)** and **Predicted (line)** once records exist.

## 5) How to push data daily
This Streamlit app *reads* from Supabase. You should push predictions daily from:
- a scheduled job (Windows Task Scheduler / cron)
- GitHub Actions
- any internal server

Minimum fields to upsert:
- `asset`
- `as_of_date`
- `target_date`
- `predicted_value`
- `actual_value` (optional but required for accuracy)
- `unit`, `model_name`, `horizon`
