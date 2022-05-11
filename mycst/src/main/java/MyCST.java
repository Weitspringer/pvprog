import com.google.api.client.auth.oauth2.Credential;
import com.google.api.client.extensions.java6.auth.oauth2.AuthorizationCodeInstalledApp;
import com.google.api.client.extensions.jetty.auth.oauth2.LocalServerReceiver;
import com.google.api.client.googleapis.auth.oauth2.GoogleAuthorizationCodeFlow;
import com.google.api.client.googleapis.auth.oauth2.GoogleClientSecrets;
import com.google.api.client.googleapis.javanet.GoogleNetHttpTransport;
import com.google.api.client.http.FileContent;
import com.google.api.client.http.javanet.NetHttpTransport;
import com.google.api.client.json.JsonFactory;
import com.google.api.client.json.gson.GsonFactory;
import com.google.api.client.util.store.FileDataStoreFactory;
import com.google.api.services.drive.Drive;
import com.google.api.services.drive.DriveScopes;
import com.google.api.services.drive.model.File;
import com.google.api.services.drive.model.Permission;

import java.awt.*;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.security.GeneralSecurityException;
import java.util.Collections;
import java.util.List;

/* class to demonstrate use of Drive files list API */
public class MyCST {
    /**
     * Application name.
     */
    private static final String APPLICATION_NAME = "MyCST";
    /**
     * Global instance of the JSON factory.
     */
    private static final JsonFactory JSON_FACTORY = GsonFactory.getDefaultInstance();
    /**
     * Directory to store authorization tokens for this application.
     */
    private static final String TOKENS_DIRECTORY_PATH = "tokens";

    /**
     * Global instance of the scopes required by this quickstart.
     * If modifying these scopes, delete your previously saved tokens/ folder.
     */
    private static final List<String> SCOPES = Collections.singletonList(DriveScopes.DRIVE_FILE);
    private static final String CREDENTIALS_FILE_PATH = "/credentials.json";
    private static final String PVPROG_FOLDER_ID = "1oBfvDi3ubrHVGTSc1egGfxv_4EljHiO2";

    /**
     * Creates an authorized Credential object.
     *
     * @param HTTP_TRANSPORT The network HTTP Transport.
     * @return An authorized Credential object.
     * @throws IOException If the credentials.json file cannot be found.
     */
    private static Credential getCredentials(final NetHttpTransport HTTP_TRANSPORT) throws IOException {
        // Load client secrets.
        InputStream in = MyCST.class.getResourceAsStream(CREDENTIALS_FILE_PATH);
        if (in == null) {
            throw new FileNotFoundException("Resource not found: " + CREDENTIALS_FILE_PATH);
        }
        GoogleClientSecrets clientSecrets = GoogleClientSecrets.load(JSON_FACTORY, new InputStreamReader(in));

        // Build flow and trigger user authorization request.
        GoogleAuthorizationCodeFlow flow = new GoogleAuthorizationCodeFlow.Builder(
                HTTP_TRANSPORT, JSON_FACTORY, clientSecrets, SCOPES)
                .setDataStoreFactory(new FileDataStoreFactory(new java.io.File(TOKENS_DIRECTORY_PATH)))
                .setAccessType("offline")
                .build();
        LocalServerReceiver receiver = new LocalServerReceiver.Builder().setPort(8888).build();
        Credential credential = new AuthorizationCodeInstalledApp(flow, receiver).authorize("user");
        //returns an authorized Credential object.
        return credential;
    }


    public static void main(String... args) throws GeneralSecurityException, IOException {
        // Build a new authorized API client service.
        final NetHttpTransport HTTP_TRANSPORT = GoogleNetHttpTransport.newTrustedTransport();
        Drive service = new Drive.Builder(HTTP_TRANSPORT, JSON_FACTORY, getCredentials(HTTP_TRANSPORT))
                .setApplicationName(APPLICATION_NAME)
                .build();

        // Choose a file through the file explorer
        FileDialog fileDialog = new FileDialog((Frame) null, "Select File to Open");
        fileDialog.setMode(FileDialog.LOAD);
        fileDialog.setVisible(true);
        String localDirectory = fileDialog.getDirectory();
        String localFile = fileDialog.getFile();

        // Upload the file
        String fileID = uploadFile(localDirectory + localFile, service);

        // Make the file publicly visible
        Permission permission = new Permission().setType("anyone").setRole("reader");
        service.permissions().create(fileID, permission).execute();

        // Print the URL of the file
        String urlSubstring = "https://drive.google.com/file/d/";
        System.out.println(urlSubstring + fileID);

        // End the Program with Status Code 0
        System.exit(0);
    }

    private static String uploadFile(String path, Drive service) {
        File fileMetadata = new File();
        fileMetadata.setName(path.substring(path.lastIndexOf(java.io.File.separatorChar) + 1));
        fileMetadata.setParents(Collections.singletonList(PVPROG_FOLDER_ID));
        java.io.File filePath = new java.io.File(path);
        if (filePath.exists()) {
            try {
                String fileType = Files.probeContentType(filePath.toPath());
                FileContent mediaContent = new FileContent(fileType, filePath);
                File file = service.files().create(fileMetadata, mediaContent)
                        .setFields("id")
                        .execute();
                System.out.println("Upload successful!");
                return file.getId();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        } else System.out.println("File does not exist.");
        return null;
    }
}