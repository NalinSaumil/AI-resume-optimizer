from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint = 'https://dg1-sn.openai.azure.com/',
    api_key = 'e135ffd6b14f4957a3b94ac1c8ba91d4',
    api_version = '2024-02-01'
)

deployment_name = 'DG1-SN-GPT-35-TURBO'

prompt = [
    {'role':'system', 'content':'You take resume text and return modified resume text to be optimized for a specified job position.'},
    {'role':'user', 'content':'Here is my resume: Saumil Nalin McKinney, Texas, 75072   |   +1 (860) 830-6331   |    Saumiln@gmail.com www.linkedin.com/in/saumil-nalin-3705-cpmaj Profile Summary: As an IT enthusiast, I have been trained in AWS Cloud, Google AI, Azure OpenAI Services and Python/Java coding. In college, I led projects like Automation via OpenAI API and Thermoacoustic lab experimentation. I was also the MATLAB Student Ambassador at MathWorks, deeply involved with Physics, CS and Mechanical Engineering students at the University of Texas at Dallas. I am seeking an opportunity where I can combine my technical skills with effective communication and strong leadership to meet the organizational goals and progress in my career path. I bring fresh insights in AI applications, captivated by the transformative potential of the latest technologies. Education: Bachelor of Science: Physics and Computer Science	Dec 2023 The University of Texas at Dallas	GPA: 3.34 Academic Excellence Scholarship Skills: - Java and Python Programming - Artificial Intelligence		                      - Web Development - AWS Cloud Services			            - Microsoft Office Suite		                      - CAD - MATLAB			                          - Terraform & Ansible	                                    - Effective Communication - Problem Solving                                                     - Team Leadership Certifications: - Google Analytics                         - Google Cloud Generative AI Fundamentals                     - AWS Certified Cloud Practitioner - MATLAB Fundamentals             - LPS Qubit Collaboratory Summer of Quantum               - Generative AI solutions with Azure Projects: Storybook Generator AI	May 2023 – May 2023 -	Utilized Python with OpenAI and StreamLit APIs to develop an AI frontend that generates a picture book from a given title including up to 5-pictures, and an audio recording to narrate the story. Speech to Image Converter	Apr 2023 – Apr 2023 -	Employed OpenAI API to convert audio to image using speech-to-text for transcription and integrated that into image generation. ECS Chatbot	Feb 2023 – Apr 2023 -	Built OpenAI-powered chatbot to aid engineering campus advisors with student FAQs. -	Utilized extractive AI on department website PDF data obtained via web scraping, reducing advisor workload. Study on the Thermoacoustic Effect	Jan 2023 – May 2023 -	Led a 5-month thermoacoustic research project, involving CAD-based stack design, complex electronics manipulation, instrumentation, and successful recreation of thermoacoustic effects, demonstrating strong teamwork and technical expertise in experimental design. Work Experience: MathWorks, MATLAB Student Ambassador 	Mar 2023 – Dec 2023 -	Explained a crucial engineering resource to assist over 100 students in connecting with engineering interests. -	Coordinated social media communications with over 300% increase in account reach and interaction in 9 months. Kappa Theta Pi Fraternity – Mu Colony, Back-End Developer 	Feb 2023 – Aug 2023 -	Built OpenAI-powered chatbot to aid engineering campus advisors’ load with over 200 tested student FAQs. -	Utilized extractive AI on department website 87-page PDF data obtained via web scraping. Society of Physics Students, Treasurer 	Aug 2020 – Dec 2023 -	Financed a 501(c)(3) organization established to promote physics on campus with over 50 members. -	Coordinated with other officers to hold 2-3 events per semester and gain student reach of over 200%.; modify it to be optimized for a enrty level Cloud engineer position.'}
]

response = client.chat.completions.create(model=deployment_name, messages=prompt)

print(response.choices[0].message.content)