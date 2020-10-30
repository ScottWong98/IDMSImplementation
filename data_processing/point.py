class Point:

    def __init__(self, user_id, time_in, time_out, longitude,
                 latitude, duration, day_number, total_data):
        self.user_id = user_id
        self.time_in = time_in
        self.time_out = time_out
        self.longitude = longitude
        self.latitude = latitude
        self.duration = duration
        self.day_number = day_number
        self.total_data = total_data
        self.sr = 0

    def merge(self):
        pass

    def remove(self):
        pass

    def sr2cluster(self, cluster):
        self.sr = cluster

    def sr2semantics(self, poi):
        self.sr = poi

    def __str__(self) -> str:
        return f"{self.user_id}\t{self.time_in}\t{self.time_out}\t{self.longitude}\t{self.latitude}\t{self.duration}" \
               f"\t{self.day_number}\t{self.total_data}\t{self.sr}"
