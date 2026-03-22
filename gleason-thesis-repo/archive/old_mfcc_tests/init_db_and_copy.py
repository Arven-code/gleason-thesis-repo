from sqlalchemy import create_engine, Column, Integer, String, text
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class WordInterval(Base):
    __tablename__ = 'word_interval'
    id = Column(Integer, primary_key=True)
    word = Column(String)
    start_time = Column(Integer)
    end_time = Column(Integer)

class WordIntervalTemp(Base):
    __tablename__ = 'word_interval_temp'
    id = Column(Integer, primary_key=True)
    word = Column(String)
    start_time = Column(Integer)
    end_time = Column(Integer)

engine = create_engine("sqlite:///my_database.db", echo=True)
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

temp_entry1 = WordIntervalTemp(word="hello", start_time=1000, end_time=2000)
temp_entry2 = WordIntervalTemp(word="world", start_time=3000, end_time=4000)
session.add_all([temp_entry1, temp_entry2])
session.commit()
print("Data inserted into word_interval_temp.")

with engine.begin() as conn:
    conn.execute(
        text("INSERT INTO word_interval (word, start_time, end_time) SELECT word, start_time, end_time FROM word_interval_temp")
    )
print("Data copied to word_interval without conflicting IDs.")

print("Permanent table contents:")
for row in session.query(WordInterval).all():
    print(f"ID: {row.id}, Word: {row.word}, Start: {row.start_time}, End: {row.end_time}")
