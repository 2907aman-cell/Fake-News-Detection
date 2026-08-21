"""
Dataset Module for Fake News Detector.
Provides initial benchmark dataset combining standard news items (REAL)
and sensationalist/fabricated news items (FAKE) across multiple domains.
"""

import pandas as pd
import numpy as np

INITIAL_DATASET = [
    # REAL NEWS SAMPLES (Label: 1)
    {
        "title": "Federal Reserve Holds Interest Rates Steady Amid Economic Indicators",
        "text": "The Federal Reserve announced on Wednesday that it will maintain current benchmark interest rates following a two-day meeting of policy makers. Officials cited steady employment growth and moderate inflation metrics as primary reasons for keeping rates unchanged. Economic analysts noted that the decision reflects a cautious approach to monetary policy in light of shifting global trade market conditions.",
        "label": 1
    },
    {
        "title": "NASA Rover Discovers Complex Organic Molecules on Mars Surface",
        "text": "Scientists analyzing data returned by NASA's Mars Perseverance rover have identified complex carbon-based molecules in sediment samples collected from an ancient river delta. Researchers clarified that while organic molecules are essential building blocks for life, their presence does not definitively prove past microbial life on the Red Planet without further laboratory examination of returned samples.",
        "label": 1
    },
    {
        "title": "Global Climate Summit Concludes with Renewable Energy Investment Pact",
        "text": "Representatives from over one hundred nations concluded the international climate conference by signing a binding accord to accelerate renewable energy deployment. The agreement mandates targeted reductions in greenhouse gas emissions over the next decade and establishes a dedicated multilateral clean energy transition infrastructure fund.",
        "label": 1
    },
    {
        "title": "European Union Passes Comprehensive Artificial Intelligence Regulations",
        "text": "European lawmakers voted overwhelmingly to adopt the landmark Artificial Intelligence Act, establishing a risk-based regulatory framework for AI systems. High-risk applications such as biometric surveillance and automated hiring algorithms face stringent compliance requirements and transparency mandates designed to safeguard consumer privacy.",
        "label": 1
    },
    {
        "title": "Tech Giant Unveils Energy-Efficient Quantum Computing Processor",
        "text": "Engineers today demonstrated a new 100-qubit quantum processor featuring significantly reduced error rates and lower operational cooling power. The groundbreaking hardware architecture allows researchers to perform complex molecular simulations in minutes that previously required months on conventional supercomputers.",
        "label": 1
    },
    {
        "title": "World Health Organization Reports Decline in Global Transmission Rates",
        "text": "Epidemiologists at the World Health Organization published new surveillance data indicating a sustained decrease in seasonal viral transmission rates worldwide. Health authorities attributed the trend to increased vaccination coverage, improved ventilation in public spaces, and community health initiatives.",
        "label": 1
    },
    {
        "title": "Central Bank Digital Currency Trial Successfully Completes First Phase",
        "text": "The financial monetary authority completed the initial pilot phase of its sovereign digital currency, processing over one million test transactions across commercial retail banks. The evaluation demonstrated strong network stability, low transaction latency, and robust encryption protocols.",
        "label": 1
    },
    {
        "title": "Astronomers Detect Exoplanet with Water Vapor in Atmosphere",
        "text": "Using space telescope spectroscopic analysis, astronomical researchers confirmed the detection of water vapor in the atmosphere of a habitable-zone exoplanet located 120 light years away. Further observations will measure atmospheric pressure and chemical composition.",
        "label": 1
    },
    {
        "title": "Major Infrastructure Bill Passed to Repair Bridges and Transit Systems",
        "text": "Parliament approved bipartisan legislation designating funds for national highway reconstruction, bridge repairs, and high-speed rail network modernization. Project construction contracts are scheduled to begin early next year following environmental reviews.",
        "label": 1
    },
    {
        "title": "Medical Trial Demonstrates Success of Novel Targeted Cancer Immunotherapy",
        "text": "Phase III clinical trial results published in the Journal of Oncology demonstrate significant tumor reduction in patients receiving a personalized mRNA cancer vaccine. Medical researchers highlighted high treatment tolerance and prolonged progression-free survival rates.",
        "label": 1
    },
    {
        "title": "Automated Solar Panel Manufacturing Plant Opens in Midwest Region",
        "text": "A state-of-the-art clean energy manufacturing facility opened today, capable of producing high-efficiency photovoltaic panels annually. The initiative created twelve hundred clean technology jobs and expanded local industrial renewable capacity.",
        "label": 1
    },
    {
        "title": "International Trade Agreement Reduces Tariffs on Agricultural Goods",
        "text": "Trade ministers signed a bilateral agreement eliminating import duties on key agricultural exports, expanding market access for domestic farmers and lowering consumer food prices across participating nations.",
        "label": 1
    },
    {
        "title": "Deep Sea Marine Conservation Area Established by Coastal Alliance",
        "text": "Environmental ministries formed a multi-nation ocean sanctuary protecting fragile coral reef ecosystems and marine species from commercial deep-sea mining and industrial trawling activities.",
        "label": 1
    },
    {
        "title": "New High-Density Battery Chemistry Extends Electric Vehicle Range",
        "text": "Materials scientists developed a solid-state lithium battery cell offering twice the energy density of conventional lithium-ion batteries. The breakthrough promises extended driving range and significantly faster charging cycles.",
        "label": 1
    },

    # FAKE NEWS SAMPLES (Label: 0)
    {
        "title": "SHOCKING SECRET: Miracle Herb Instantly Cures All Known Diseases Overnight!",
        "text": "Big Pharma elites don't want you to know this secret! A hidden miraculous plant discovered in the Amazon jungle completely eradicates all cancers, diabetes, and heart disease in less than 24 hours! Government regulators banned this cure to protect corporate profits! Doctors are furious that this simple home remedy is leaking online! Click now before this video is deleted forever!",
        "label": 0
    },
    {
        "title": "CONFIRMED: Government Installing Mind Control Microchips in Municipal Water Supply!",
        "text": "Unbelievable leaked documents prove beyond any doubt that secretive global cabals are adding nanoscale tracking microchips into tap water nationwide! Whistleblowers reveal that these microscopic devices connect directly to 5G towers to control human thought patterns and alter voter behavior. SHARE THIS URGENT WARNING WITH EVERYONE YOU KNOW BEFORE IT IS WIPED FROM THE INTERNET!",
        "label": 0
    },
    {
        "title": "Hollywood Celebrity Secretly Replaced by Biological Clone after Mysterious Disappearance",
        "text": "Insiders confirm that famous A-list actor was covertly substituted by a genetically engineered clone following a top-secret offshore incident! High-level whistleblowers claim the clone failed public voice stress tests. Photos don't lie! Look at the earlobes in these side-by-side images! The mainstream media is covering up the truth!",
        "label": 0
    },
    {
        "title": "ALERT: Earth to Experience 15 Days of Total Darkness Next Month Says Banned Scientist",
        "text": "A world-renowned astronomer who was fired by space agencies for telling the truth has warned that a massive celestial shadow will plunge the entire planet into pitch darkness for two whole weeks! Power grids will collapse, magnetic poles will flip, and governments are building underground bunkers for the elite while hiding the truth from ordinary citizens!",
        "label": 0
    },
    {
        "title": "BANNED FOOD: Common Grocery Item Proven to Cause Instant Memory Loss!",
        "text": "You won't believe what they are putting in your cereal! Independent rogue researchers discovered that a common food additive is causing mass amnesia and brain shrinkage! Food safety agencies were bribed millions of dollars to keep quiet! Throw out this food item immediately before it damages your family's health!",
        "label": 0
    },
    {
        "title": "Billionaire Secretly Buys Entire Weather System to Trigger Controlled Storms",
        "text": "Shocking investigative report exposes how tech billionaires bought orbital geoengineering satellites to manipulate weather patterns! Insiders reveal artificial hurricanes are being generated to manipulate real estate prices and control global agricultural yields. They don't want you to see this evidence!",
        "label": 0
    },
    {
        "title": "Ancient Alien Pyramid Uncovered Beneath Antarctica Ice Sheet Proves Forgotten Empire",
        "text": "Satellite imagery reveals a massive metallic pyramid hidden under Antarctic ice! Leaked military memos reveal elite special forces were dispatched to retrieve ancient alien technology capable of unlimited zero-point energy! Mainstream news outlets refuse to report this historic discovery!",
        "label": 0
    },
    {
        "title": "SECRET AGENDA: Secret Law Passed Overnight Eliminating All Bank Deposits!",
        "text": "Urgent financial alert! Emergency legislation passed in the dead of night grants central bankers authority to confiscate all personal savings accounts tomorrow morning! Financial experts urge everyone to immediately withdraw all cash and buy gold before the global financial shutdown occurs!",
        "label": 0
    },
    {
        "title": "PROVED BEYOND DOUBT: Popular Smartphone App Secretly Records Audio While Turned Off",
        "text": "Rogue cybersecurity experts caught top social media apps secretly transmitting continuous microphone recordings to hidden offshore servers even when devices are fully powered down! Millions of users are being spied on 24/7! Delete this app right now!",
        "label": 0
    },
    {
        "title": "Time Traveler Arrives from Year 2090 with Shocking Proof of Upcoming Events",
        "text": "Self-proclaimed time traveler passed a polygraph test revealing precise details about future historical events! Video evidence shows inexplicable futuristic gadgets and predictions about global stock market crashes happening next week. Watch the full interview before officials take it down!",
        "label": 0
    },
    {
        "title": "MIRACLE FRUIT Kills Cancer Cells 10,000 Times Stronger Than Chemotherapy!",
        "text": "Laboratory studies suppressed by pharmaceutical conglomerates prove that an exotic tropical fruit destroys malignant tumors effortlessly without side effects! Big Pharma spends billions trying to destroy seeds of this plant! Share this life-saving secret with your loved ones!",
        "label": 0
    },
    {
        "title": "LEAKED AUDIO: Politicians Caught Scheming to Ban Private Vehicle Ownership",
        "text": "Secret wiretap recording leaks global political leaders discussing secret plans to confiscate private automobiles and force citizens into centralized surveillance zones! Authorities claim audio is fake, but audio experts confirm it is 100 percent authentic!",
        "label": 0
    },
    {
        "title": "Mysterious Energy Sphere Discovered in Deep Ocean Defies All Laws of Physics",
        "text": "Naval divers operating at extreme ocean depths discovered a glowing sphere emitting infinite clean energy that burns under water! Defense contractors immediately classified the site and placed the ocean zone under military blockade to keep the technology secret from the public!",
        "label": 0
    },
    {
        "title": "WARNING: Household Microwave Ovens Alter DNA Structure in Cooked Food",
        "text": "Sensational medical discovery proves that microwave radiation mutates food molecules into toxic carcinogens that permanently alter human cellular DNA! Switch to raw organic foods immediately to reverse the damage caused by years of microwave usage!",
        "label": 0
    }
]

def load_initial_dataset():
    """Returns dataset as a pandas DataFrame."""
    df = pd.DataFrame(INITIAL_DATASET)
    # Combine title and text for training
    df['full_text'] = df['title'] + " " + df['text']
    return df

def get_dataset_stats(df):
    """Returns dataset statistics."""
    total = len(df)
    real_count = int((df['label'] == 1).sum())
    fake_count = int((df['label'] == 0).sum())
    return {
        "total_samples": total,
        "real_count": real_count,
        "fake_count": fake_count,
        "real_percentage": round((real_count / total) * 100, 1) if total > 0 else 0,
        "fake_percentage": round((fake_count / total) * 100, 1) if total > 0 else 0
    }
