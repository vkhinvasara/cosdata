use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use lmdb::{Database, Environment, Transaction, WriteFlags};
use serde_cbor::{from_slice, to_vec};
use crate::models::common::WaCustomError;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct User {
    pub user_id: u32,
    pub username: String,
    pub password_hash: String,
    pub roles: HashSet<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Hash)]
pub enum Permission {
    ListCollections,
    CreateCollection,
    DeleteCollection,
    ListIndex,
    CreateIndex,
    DeleteIndex,
    UpsertVectors,
    DeleteVectors,
    QueryVectors,
    ListVersions,
    SetCurrentVersion,
    GetCurrentVersion,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Role {
    pub role_id: u32,
    pub role_name: String,
    pub description: String,
    pub permissions: Vec<(u32, Permission)>, // (collection_id, permission) pairs
}

#[derive(Debug)]
pub struct RbacManager {
    env: Environment,
    users_db: Database,
    roles_db: Database,
    counters_db: Database,
}

impl RbacManager {
    pub fn new(env: Environment) -> Result<Self, WaCustomError> {
        let txn = env.begin_rw_txn()?;
        
        // Create databases if they don't exist
        let users_db = unsafe { 
            txn.create_db(Some("users"), lmdb::DatabaseFlags::empty())?
        };
        let roles_db = unsafe { 
            txn.create_db(Some("roles"), lmdb::DatabaseFlags::empty())?
        };
        let counters_db = unsafe { 
            txn.create_db(Some("counters"), lmdb::DatabaseFlags::empty())?
        };
        
        txn.commit()?;

        Ok(Self {
            env,
            users_db,
            roles_db,
            counters_db,
        })
    }

    fn get_next_id(&self, counter_key: &str) -> Result<u32, lmdb::Error> {
        let mut txn = self.env.begin_rw_txn()?;
        let current_id = match txn.get(self.counters_db, &counter_key) {
            Ok(id) => id,
            Err(e) => return Err(e),
        };
        
        let next_id = u32::from_be_bytes(current_id.try_into().unwrap()) + 1;
        txn.put(
            self.counters_db,
            &counter_key,
            &next_id.to_be_bytes(),
            WriteFlags::empty(),
        )?;
        
        txn.commit()?;
        Ok(next_id)
    }

    pub fn create_user(&self, username: &str, password: &str) -> Result<User, WaCustomError> {
        let user_id = self.get_next_id("user_counter")?;
        let password_hash = sha256_hash(password);

        let user = User {
            user_id,
            username: username.to_string(),
            password_hash,
            roles: HashSet::new(),
        };

        let key = format!("user:{}", user_id);
        let value = to_vec(&user)?;

        let mut txn = self.env.begin_rw_txn()?;
        txn.put(self.users_db, &key, &value, WriteFlags::empty())?;
        txn.commit()?;

        Ok(user)
    }

    pub fn create_role(&self, name: &str, description: &str) -> Result<Role, WaCustomError> {
        let role_id = self.get_next_id("role_counter")?;

        let role = Role {
            role_id,
            role_name: name.to_string(),
            description: description.to_string(),
            permissions: Vec::new(),
        };

        let key = format!("role:{}", role_id);
        let value = to_vec(&role)?;

        let mut txn = self.env.begin_rw_txn()?;
        txn.put(self.roles_db, &key, &value, WriteFlags::empty())?;
        txn.commit()?;

        Ok(role)
    }

    pub fn assign_role_to_user(&self, user_id: u32, role_id: u32) -> Result<(), WaCustomError> {
        let key = format!("user:{}", user_id);
        let mut txn = self.env.begin_rw_txn()?;
        
        let user_data = txn.get(self.users_db, &key)?;
        let mut user: User = from_slice(user_data)?;
        
        user.roles.insert(role_id);
        
        let value = to_vec(&user)?;
        txn.put(self.users_db, &key, &value, WriteFlags::empty())?;
        txn.commit()?;
        
        Ok(())
    }

    pub fn add_permission_to_role(
        &self,
        role_id: u32,
        collection_id: u32,
        permission: Permission,
    ) -> Result<(), WaCustomError> {
        let key = format!("role:{}", role_id);
        let mut txn = self.env.begin_rw_txn()?;
        
        let role_data = txn.get(self.roles_db, &key)?;
        let mut role: Role = from_slice(role_data)?;
        
        role.permissions.push((collection_id, permission));
        
        let value = to_vec(&role)?;
        txn.put(self.roles_db, &key, &value, WriteFlags::empty())?;
        txn.commit()?;
        
        Ok(())
    }

    pub fn check_permission(
        &self,
        user_id: u32,
        collection_id: u32,
        required_permission: &Permission,
    ) -> Result<bool, WaCustomError> {
        let txn = self.env.begin_ro_txn()?;
        
        let user_key = format!("user:{}", user_id);
        let user_data = txn.get(self.users_db, &user_key)?;
        let user: User = from_slice(user_data)?;
        
        for role_id in user.roles {
            let role_key = format!("role:{}", role_id);
            let role_data = txn.get(self.roles_db, &role_key)?;
            let role: Role = from_slice(role_data)?;
            
            if role.permissions.iter().any(|(cid, perm)| 
                *cid == collection_id && perm == required_permission
            ) {
                return Ok(true);
            }
        }
        
        Ok(false)
    }
}

fn sha256_hash(input: &str) -> String {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    format!("{:x}", hasher.finalize())
}
