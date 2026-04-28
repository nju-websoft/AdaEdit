source_code = """
def process_user_batch(users):
    for user in users:
        if not check_active(user):
            print('Warning: Inactive user')
            process_user(user)
            continue

        if user.age >= 16:
            process_user(user)
""".strip() 

target_code = """
def process_user_batch(users):
    for user in users:
        if not check_active(user):
            print('Warning: Inactive user')
            process_user(user, active=False)
            continue

        if user.age >= 16:
            process_user(user)
""".strip()
