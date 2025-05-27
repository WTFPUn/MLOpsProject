from prisma import Prisma
from prisma.models import NewsTest

from datetime import datetime, date
from datetime import timedelta


def get_start_of_week(date: datetime) -> tuple:
    # Get the start of the week (Monday)
    start_of_week = date - timedelta(days=date.weekday())
    
    # return only the date part
    return start_of_week.date()

async def seed():
    db = Prisma(auto_register=True)
    await db.connect()
    
    # Delete all existing news articles
    await db.news.delete_many()
     
    start_date_last = datetime.combine(get_start_of_week(datetime(2025,5,22)), datetime.min.time())
    # first is before last 2 months
    start_date_first = datetime.combine(get_start_of_week(datetime(2025,4,1)), datetime.min.time())

    # Create a test news article
    test_news = await db.news.create_many(
        data=[
             {
            "title": "First News cluster 0",
            "content": "This is a test news article.",
            "startDate": start_date_first,
            "cluster": 0,
        },
        {
            "title": "Lastest News cluster 0",
            "content": "This is a test news article.",
            "startDate": start_date_last,
            "cluster": 0,
        },
        {
            "title": "Latest News cluster 1",
            "content": """
            **สรุปข่าว:**
ณิชา ณัฏฐณิชา ห่างจาก โตโน่ ภาคิน และไปคบกับผู้ช่วยผู้จัดการส่วนตัวของตนเอง แต่ โตโน่ ได้ออกมาขอโทษแฟนคลับที่เกิดความไม่ดี และบอกว่าจะกลับไปอยู่นาเหมือนเดิม

**ไทม์ไลน์เหตุการณ์:**
*   **ปลายเดือนมีนาคม:** ณิชา และ โตโน่ ได้ไปเยี่ยมครอบครัวของ ณิชา ที่เชียงใหม่และได้อุ้มหลานของ ณิชา
*   **12 เมษายน:** ณิชา ได้เคลื่อนไหวโพสต์ข้อความผ่านไอจีสตอรี่ โดยเธอลงภาพท้องฟ้าสีส้มสวยงามในยามเย็น
*   **13 เมษายน:** โตโน่ ได้เล่นคอนเสิร์ตที่ สุโขทัย และร้องเพลงขอโทษแฟนเพลง
*   **13 เมษายน:** โตโน่ ได้พูดถึงความสัมพันธ์กับ ณิชา และผู้ช่วยผู้จัดการส่วนตัว ผ่านคลิปเสียง
*   **13 เมษายน:** ผู้จัดการส่วนตัวของ โตโน่ เผยว่า โตโน่ สนิทกับฝ่ายหญิงในช่วง 2 อาทิตย์ที่ผ่านมา แต่พอรู้ว่าฝ่ายหญิงมีแฟนแล้วก็เลยขอยุติความสัมพันธ์ทุกอย่างลงในทันที"
            """,
            "startDate": start_date_last,
            "cluster": 1,
        },
        {
            "title": "Latest News cluster 2",
            "content": """
            **สรุปข่าว:**
กลุ่มข่าวนี้เกี่ยวกับการหลอกลวงของ ""ไฮโซฮอต"" (นายธัญเทพ ศิริทรัพย์เดชากุล) ที่ใช้ภาพถ่ายและข้อมูลปลอมเพื่อสร้างความน่าเชื่อถือและหลอกลวงผู้หญิงหลายคน รวมถึง ""คะน้า"" ที่ตกหลุมรักและเกือบแต่งงานด้วย นอกจากนี้ ยังมีข้อมูลว่า ""ฮอต"" มีลูกด้วย แต่ไม่ยอมรับและหลีกเลี่ยงการรับผิดชอบ รวมถึงมีผู้เสียหายอีกเกือบ 10 คน และกองปราบปรามได้ค้นบ้านพักของ ""ฮอต"" ยึดหลักฐานจำนวน 23 รายการ และกองทัพอากาศกำลังตรวจสอบเรื่องที่ ""ฮอต"" มีความเกี่ยวข้องกับทหารยศพันเอก

**ไทม์ไลน์เหตุการณ์:**
*   **11 เมษายน 2568:** กัน จอมพลัง เปิดเผยว่ามีผู้เสียหายออกมาให้ข้อมูลแล้วเกือบ 10 คน และกำลังจะพาผู้เสียหายไปแจ้งความที่กองบัญชาการตำรวจสอบสวนกลาง
*   **11 เมษายน 2568:** กัน จอมพลัง ได้รับสายปริศนาโทรมาหา ""คะน้า"" ขอให้ถอนแจ้งความ
*   **11 เมษายน 2568:** กัน จอมพลัง ส่งเรื่องให้กองทัพอากาศตรวจสอบเรื่องเข็มกลัดผู้คุณทำประโยชน์ต่อกองทัพอากาศ
*   **13 เมษายน 2568:** กองปราบปรามเข้าตรวจค้นบ้านพัก ""ไฮโซฮอต"" ยึดหลักฐาน 23 รายการ
*   **10 เมษายน 2568:** ""ฮอต"" อ้างว่ามีลูกไม่ได้เพราะเป็นโรคลูคีเมียและธาลัสซีเมีย และหลีกเลี่ยงการรับผิดชอบ
*   **เมษายน 2565:** ""ฮอต"" เริ่มติดต่อ ""คะน้า"" ผ่านแอปหาคู่ และเริ่มจีบ
*   **เมษายน 2565:** ""ฮอต"" ชวน ""คะน้า"" ไปหา และเสนอว่าจะโอนเงินค่าเดินทาง
*   **เมษายน 2565:** ""คะน้า"" ตกลงคบหาดูใจกับ ""ฮอต""
*   **มิถุนายน 2565:** ""ฮอต"" เริ่มพูดคุยน้อยลง และ ""คะน้า"" ค้นพบว่าตัวเองท้อง
*   **ธันวาคม 2565:** ""คะน้า"" คลอดลูก แต่ ""ฮอต"" ไม่รับผิดชอบ
*   **เมษายน 2568:** เรื่องราวถูกเปิดเผยและมีการสืบสวนเพิ่มเติม"
""",
            "startDate": start_date_last,
            "cluster": 2,
        },
        {
            "title": "Latest News cluster 3",
            "content": "This is a test news article of cluster 3.",
            "startDate": start_date_last,
            "cluster": 3,
        },
        {
            "title": "Latest News cluster 4",
            "content": "This is a test news article of cluster 4.",
            "startDate": start_date_last,
            "cluster": 4,
        },
        {
            "title": "Latest News cluster 5",
            "content": "This is a test news article of cluster 5.",
            "startDate": start_date_last,
            "cluster": 5,
        },
        ],              
    )

    print(f"Created test news article: {test_news}")

    await db.disconnect()
    
    
if __name__ == "__main__":
    import asyncio
    # print(get_start_of_week(datetime(2025,5,22)))
    asyncio.run(seed())