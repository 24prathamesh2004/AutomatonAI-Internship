
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import certifi
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.server_api import ServerApi
import pytz

_CLIENT_INSTANCE: Optional[MongoClient] = None

def get_mongo_uri() -> str:
    try:
        from dotenv import load_dotenv
        _env_path = Path(__file__).resolve().parent / ".env"
        load_dotenv(dotenv_path=_env_path)
    except ImportError:
        pass
    uri = os.environ.get("MONGODB_URI", "")
    if not uri:
        raise ValueError(
            "MONGODB_URI not set. Set it in .env for local dev or in HF Space secrets."
        )
    return uri


def get_db() -> Database:
    global _CLIENT_INSTANCE
    
    if _CLIENT_INSTANCE is not None:
        return _CLIENT_INSTANCE.get_database("attendance_db")

    uri = get_mongo_uri()
    
    tls_opts = {
        "tlsCAFile": certifi.where(),
        "server_api": ServerApi("1"),
        "serverSelectionTimeoutMS": 5000,
        "connectTimeoutMS": 10000,
        "maxPoolSize": 1,
        "tls": True,
        "tlsAllowInvalidCertificates": True,  
        "tlsAllowInvalidHostnames": True,     
    }

    print("Connecting to MongoDB Atlas...")
    try:
        # We pass tls_opts as kwargs to MongoClient
        _CLIENT_INSTANCE = MongoClient(uri, **tls_opts)
        _CLIENT_INSTANCE.admin.command('ping')
        print("Connected to MongoDB Atlas successfully.")
        
    except Exception as e:
        print("Failed to connect to MongoDB.")
        print(f"Error details: {e}")
        # Reset client if failed so we can try again next time
        _CLIENT_INSTANCE = None
        raise e

    return _CLIENT_INSTANCE.get_database("attendance_db")
  
def get_enrollment_by_user_id(user_id: str, db: Optional[Database] = None) -> Optional[dict]:
    coll = get_enrollments_collection(db)
    return coll.find_one({"user_id": user_id}, {"name": 1, "user_id": 1}) 

def get_enrollments_collection(db: Optional[Database] = None) -> Collection:
    if db is None:
        db = get_db()
    return db["enrollments"]

def get_attendance_collection(db: Optional[Database] = None) -> Collection:
    if db is None:
        db = get_db()
    return db["attendance"]


COOLDOWN_MINUTES = 10

from datetime import datetime, timedelta
from typing import Optional, List, Tuple
# Enrollment Logic 

def insert_enrollment(
    user_id: str,
    name: str,
    embeddings: List[List[float]],  # Accepts the list of 3 embeddings
    db: Optional[Database] = None,
) -> Optional[str]:
    coll = get_enrollments_collection(db)
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    # Check if user already exists
    doc = coll.find_one({"user_id": user_id})
    
    if doc:
        # Update existing user: Append new embeddings and update name
        existing_embeddings = list(doc.get("embeddings", []))
        existing_embeddings.extend(embeddings)
        coll.update_one(
            {"user_id": user_id},
            {"$set": {"embeddings": existing_embeddings, "name": name, "updated_at": now}},
        )
        return str(doc["_id"])
    
    # Create new user record
    result = coll.insert_one({
        "user_id": user_id,
        "name": name,
        "embeddings": embeddings, 
        "created_at": now,
        "updated_at": now,
    })
    return str(result.inserted_id)

def get_all_enrollments_for_cache(db: Optional[Database] = None) -> Tuple[List, List]:
    coll = get_enrollments_collection(db)
    docs = list(coll.find({}))
    embeddings_flat = []
    labels_flat = []
    for d in docs:
        uid = d["user_id"]
        for emb in d.get("embeddings", []):
            embeddings_flat.append(emb)
            labels_flat.append(uid)
    return embeddings_flat, labels_flat

# Attendance Logic 

def insert_attendance(
    user_id: str,
    db: Optional[Database] = None,
    cooldown_minutes: int = 5,
) -> bool:
    coll = get_attendance_collection(db)
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    # Cooldown check: Prevents duplicate marking within X minutes
    if cooldown_minutes > 0:
        since = now - timedelta(minutes=cooldown_minutes)
        count = coll.count_documents(
            {"user_id": user_id, "timestamp": {"$gte": since}},
            limit=1
        )
        if count > 0:
            return False

    doc = {
        "user_id": user_id,
        "timestamp": now, 
    }
    coll.insert_one(doc)
    return True

def get_attendance(
    from_date: datetime,
    to_date: datetime,
    db: Optional[Database] = None,
) -> List[dict]:
    coll = get_attendance_collection(db)
    ist = pytz.timezone('Asia/Kolkata')
    
    from_ist = from_date.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=ist)
    to_ist = to_date.replace(hour=23, minute=59, second=59, microsecond=999999, tzinfo=ist)
    
    cursor = coll.find(
        {"timestamp": {"$gte": from_ist, "$lte": to_ist}},
        sort=[("timestamp", 1)],
    )
    records = list(cursor)
    
    enroll_coll = get_enrollments_collection(db)
    unique_user_ids = list({r["user_id"] for r in records})
    
    name_map = {}
    if unique_user_ids:
        en_docs = enroll_coll.find({"user_id": {"$in": unique_user_ids}}, {"user_id": 1, "name": 1})
        for d in en_docs:
            name_map[d["user_id"]] = d.get("name", d["user_id"])
            
    for r in records:
        r["name"] = name_map.get(r["user_id"], r["user_id"])
        
    return records
