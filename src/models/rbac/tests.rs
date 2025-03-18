#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn setup_test_env() -> (Environment, tempfile::TempDir) {
        let temp_dir = tempdir().unwrap();
        let env = Environment::new()
            .set_map_size(10 * 1024 * 1024) // 10MB
            .open(temp_dir.path())
            .unwrap();
        (env, temp_dir)
    }

    #[test]
    fn test_rbac_basic_workflow() {
        let (env, _temp_dir) = setup_test_env();
        let rbac = RbacManager::new(env).unwrap();

        // Initialize admin
        let admin = rbac.initialize_admin("admin_password").unwrap();

        // Create a regular user
        let user = rbac.create_user("test_user", "password123").unwrap();

        // Create a role with limited permissions
        let role = rbac.create_role(
            "reader",
            "Can only read from collections"
        ).unwrap();

        // Add read permission to role
        rbac.add_permission_to_role(
            role.role_id,
            1, // collection_id
            Permission::QueryVectors
        ).unwrap();

        // Assign role to user
        rbac.assign_role_to_user(user.user_id, role.role_id).unwrap();

        // Check permissions
        assert!(rbac.check_permission(
            user.user_id,
            1,
            &Permission::QueryVectors
        ).unwrap());

        assert!(!rbac.check_permission(
            user.user_id,
            1,
            &Permission::CreateCollection
        ).unwrap());

        // Admin should have all permissions
        assert!(rbac.check_permission(
            admin.user_id,
            0,
            &Permission::CreateCollection
        ).unwrap());
    }
}