# Welcome to mycst!

This is a minimal cloud storage tool, where you can upload a user-specified file, automatically configure it to be world-viewable. The program will print out an url, where the uploaded file can be found. It requires Java 17 to run. Simpy execute one of the files located in the bin folder to start. The project can be built using Gradle.

Mycst is using Google Cloud to access the Google Drive API. The credentials for accessing the project are secret and located in a "credentials.json" file. Instead of the actual credentials a template file is given. To execute the program, remove the "-TEMPLATE" suffix and fill it with the actual credentials to the project. You can generate a "credentials.json" file automatically by adding an "OAuth 2.0-Client-ID" inside your Google Cloud Platform console.

Internally the programm uses your google account to upload any provided file into your personal Google Drive. To upload the file into a specific folder, create it using the traditional Google Drive interface. Copy the ID of the folder (the last part of the URL) and paste it into the "PVPROG_FOLDER_ID" constant inside the "MyCST" class.

If you want to test the program yourself please send a mail to Finn.Kaiser@student.hpi.uni-potsdam.de to be added to the group of testers.
