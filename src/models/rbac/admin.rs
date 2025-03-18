use super::*;

impl RbacManager {
    pub fn initialize_admin(&self, admin_password: &str) -> Result<User, WaCustomError> {
        // Create admin role with all permissions
        let admin_role = self.create_role(
            "admin",
            "Administrator with full system access"
        )?;

        // Add all permissions for collection ID 0 (system-wide)
        let all_permissions = vec![
            Permission::ListCollections,
            Permission::CreateCollection,
            Permission::DeleteCollection,
            Permission::ListIndex,
            Permission::CreateIndex,
            Permission::DeleteIndex,
            Permission::UpsertVectors,
            Permission::DeleteVectors,
            Permission::QueryVectors,
            Permission::ListVersions,
            Permission::SetCurrentVersion,
            Permission::GetCurrentVersion,
        ];

        for permission in all_permissions {
            self.add_permission_to_role(admin_role.role_id, 0, permission)?;
        }

        // Create admin user
        let admin_user = self.create_user("admin", admin_password)?;
        
        // Assign admin role to admin user
        self.assign_role_to_user(admin_user.user_id, admin_role.role_id)?;

        Ok(admin_user)
    }
}