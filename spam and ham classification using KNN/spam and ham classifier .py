import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Extended dataset with 50 samples
data = {
    'text': [
        'Free money!!!', 'Hi, how are you?', 'You have won a lottery!', 'Meeting at 5 PM', 'Claim your free prize now!',
        'Exclusive deal just for you!', 'Are you available for a call?', 'Congratulations, you won!', 'Let\'s catch up soon.', 'This is not a drill!',
        'Urgent: Your account has been compromised!', 'Lunch at noon?', 'Get rich quick!', 'Can we reschedule our meeting?', 'You are a winner!',
        'Important notice about your account', 'Hello!', 'Win a brand new car!', 'Reminder: Project due tomorrow', 'Earn money from home!',
        'Your bank statement is ready', 'Good morning!', 'Act now to claim your prize!', 'Are you free this weekend?', 'Biggest sale of the year!',
        'Can you review this document?', 'Final notice: Payment due', 'See you at the meeting', 'Limited time offer!', 'Just checking in.',
        'You are pre-approved for a loan!', 'Team meeting at 10 AM', 'Claim your discount now!', 'Let\'s grab coffee soon.', 'Congratulations, you\'ve been selected!',
        'Your subscription is about to expire', 'Catch up soon?', 'Don\'t miss out on this!', 'Can you send me the report?', 'Win big with our new promotion!',
        'Your order has been shipped', 'Hey there!', 'You\'ve been chosen for a prize!', 'Meeting rescheduled', 'Exclusive offer for you!',
        'Reminder: Doctor\'s appointment', 'Free gift inside!', 'Lunch plans?', 'You are eligible for a reward!', 'lets go brother'
    ],
    'label': [
        'spam', 'ham', 'spam', 'ham', 'spam',
        'spam', 'ham', 'spam', 'ham', 'spam',
        'spam', 'ham', 'spam', 'ham', 'spam',
        'spam', 'ham', 'spam', 'ham', 'spam',
        'ham', 'spam', 'ham', 'spam', 'ham',
        'spam', 'ham', 'spam', 'ham', 'spam',
        'spam', 'ham', 'spam', 'ham', 'spam',
        'ham', 'spam', 'ham', 'spam', 'ham',
        'spam', 'ham', 'spam', 'ham', 'spam',
        'ham', 'spam', 'ham', 'spam', 'ham'
    ]
}

