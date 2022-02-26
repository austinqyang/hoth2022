const express = require("express");
const app = express();
const cors = require('cors');
const res = require("express/lib/response");
app.use(cors());
app.use(express.json());
const PORT = 8000;

emotion = "undefined"; 
image_url = "undefined";

app.get('/emotion', (request, response) => {
    response.status(200).send(emotion);
});

app.get('/emotImage', (request, response) => {
    response.status(200).send(image_url);
});

app.post('/sendEmot', (request, response) => {
    const { text } = request.body;

    if(!text){
        res.status(400).send("No emotion currently");
        return;
    }
    
    emotion = text;
});

app.post('/sendImage', (request, response) => {
    const { text } = request.body;

    if(!text){
        res.status(400).send("No image currently");
        return;
    }
    
    image_url = text;
});

app.listen(PORT, () => {
    console.log(`Server listening on port ${PORT}`)
});

