from typing import List

class User:
    def __init__(self, uid, initial_model, processing_time):
        self.uid = uid
        self.current_model = initial_model
        self.processing_time = processing_time
        self.wall_clock = processing_time

    def is_ready(self):
        return self.wall_clock == 0

    def step(self):
        self.wall_clock -= 1

    def reset(self, new_model):
        self.current_model = new_model
        self.wall_clock = self.processing_time


class AsynUserPool:
    def __init__(self, users: List[User]):
        self.users = users

    def step(self):
        for user in self.users:
            user.step()

    def reset(self, new_model):
        for user in self.users:
            if user.is_ready():
                user.reset(new_model)

    def return_ready(self):
        user_ready = []
        for user in self.users:
            if user.is_ready():
                user_ready.append(user)

        return user_ready